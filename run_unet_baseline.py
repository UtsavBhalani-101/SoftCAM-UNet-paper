#!/usr/bin/env python3
"""Run the SegRDNet (UNet) APTOS baseline and archive raw console output.

Multi-GPU: when more than one GPU is visible, the trainer is launched through
torch.distributed.run (DDP, one process per GPU). SegRDNet's forward hooks are
per-process under DDP, so multi-scale feature extraction stays consistent -
nn.DataParallel would race on the hook target. Pass --no-multi-gpu to keep a
single-GPU run. The trainer caps the per-step batch size and compensates with
gradient accumulation, so the target effective batch size is forwarded as-is.
Each epoch prints a compact summary (loss, eval metrics, val QWK/AUC, recall
by grade, predicted counts, per-epoch confusion matrix), a rolling resumable
checkpoint.pt, and the full settings incl. the seed are printed at startup.

Run from the project root (or pass --project-root) in Kaggle.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    import torch
except ImportError as exc:
    raise SystemExit("PyTorch is required to determine the available GPU count.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--data-dir",
        default="/kaggle/input/competitions/aptos2019-blindness-detection",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32, help="Target effective batch size (grad accumulation)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="/kaggle/working/segrdnet_baseline")
    parser.add_argument("--no-multi-gpu", action="store_true",
                        help="Force single-GPU training even when multiple GPUs are visible")
    # wandb / resume pass-through
    parser.add_argument("--wandb", action="store_true", help="Forward wandb logging to the trainer")
    parser.add_argument("--wandb-project", default="MICCAI", type=str)
    parser.add_argument("--wandb-entity", default=None, type=str)
    parser.add_argument("--wandb-run-name", default=None, type=str)
    parser.add_argument("--resume", default=None, type=str,
                        help="Resume the trainer from checkpoint.pt: file path or its directory")
    return parser.parse_args()


def run_and_archive(label: str, command: list[str], cwd: Path, log_path: Path) -> tuple[int, str]:
    """Stream a child training run to the notebook and a raw Markdown log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        f"# {label} baseline raw output",
        "",
        f"- Started: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- Working directory: `{cwd}`",
        f"- Command: `{shlex.join(command)}`",
        "",
        "```text",
    ]
    captured: list[str] = []
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    with log_path.open("w", encoding="utf-8", newline="") as log_file:
        log_file.write("\n".join(header) + "\n")
        log_file.flush()
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            captured.append(line)
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
        log_file.write("\n```\n")
        log_file.write(f"\nExit code: {return_code}\n")

    return return_code, "".join(captured)


def extract_best_qwk(output: str) -> str:
    matches = re.findall(r"Best validation QWK:\s*([0-9.]+)\s+at epoch\s+(\d+)", output)
    return f"{matches[-1][0]} (epoch {matches[-1][1]})" if matches else "not found"


def find_unet_script(root: Path) -> Path:
    repo_names = ["unet repo", "unet_repo"]
    unet_project = "Beyond-Classification-Elaborating-Network-Predictions-for-Better-Weakly-Supervised-Quantization"
    candidates = [
        root / repo_name / unet_project / "train_scripts" / "train_aptos.py"
        for repo_name in repo_names
    ]
    script = next((path for path in candidates if path.is_file()), None)
    if script is None:
        raise SystemExit("Could not locate UNet train_scripts/train_aptos.py under the project root.")
    return script


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    unet_script = find_unet_script(root)
    unet_dir = unet_script.parent

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_ddp = gpu_count > 1 and not args.no_multi_gpu
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    archive_dir = Path(args.output) / "baseline_raw_logs" / run_id

    interpreter = sys.executable or "python"
    command = []
    if use_ddp:
        command.extend([interpreter, "-m", "torch.distributed.run",
                        "--standalone", f"--nproc_per_node={gpu_count}"])
    command.append(str(unet_script))
    command.extend([
        "--data-dir", args.data_dir,
        "--epochs", str(args.epochs),
        "--workers", str(args.workers),
        "--img-size", str(args.img_size),
        "--seed", str(args.seed),
        "--batch-size", str(args.batch_size),
        "--output", str(Path(args.output) / "segrdnet"),
    ])
    if args.resume:
        command.extend(["--resume", args.resume])
    if args.wandb:
        command.append("--wandb")
        if args.wandb_project:
            command.extend(["--wandb-project", args.wandb_project])
        if args.wandb_entity:
            command.extend(["--wandb-entity", args.wandb_entity])
        if args.wandb_run_name:
            command.extend(["--wandb-run-name", args.wandb_run_name])

    print("Baseline settings:")
    print("  Model: SegRDNet (UNet) baseline, SoftCAM Decoder 0 head disabled")
    print("  Epochs:", args.epochs)
    print("  Seed:", args.seed)
    print(f"  Target effective batch size: {args.batch_size}")
    print(f"  GPU count: {gpu_count}; training "
          + (f"multi-GPU via DDP ({gpu_count} processes)" if use_ddp else "single-GPU"))
    if args.wandb:
        print(f"  wandb: on, project {args.wandb_project}"
              + (f" (entity {args.wandb_entity})" if args.wandb_entity else ""))
    else:
        print("  wandb: off")
    if args.resume:
        print(f"  Resuming from: {args.resume}")
    print(f"  Raw Markdown log: {archive_dir / 'unet.md'}")

    print(f"\n{'=' * 72}\nStarting SegRDNet (UNet)\n{'=' * 72}")
    return_code, output = run_and_archive(
        "SegRDNet (UNet) vanilla", command, unet_dir, archive_dir / "unet.md"
    )

    summary_path = archive_dir / "baseline_summary.md"
    with summary_path.open("w", encoding="utf-8") as summary:
        summary.write("# SegRDNet (UNet) APTOS baseline\n\n")
        summary.write(f"- Seed: {args.seed}\n- Epochs: {args.epochs}\n")
        summary.write(f"- Network: SegRDNet (RDNet Tiny backbone)\n")
        summary.write(f"- Target effective batch size: {args.batch_size}\n")
        summary.write(f"- wandb: {'on, project ' + args.wandb_project if args.wandb else 'off'}\n")
        if args.resume:
            summary.write(f"- Resumed from: {args.resume}\n")
        summary.write("- SoftCAM Decoder 0 head: disabled (baseline)\n\n")
        summary.write("| Model | Exit code | Best validation QWK | Raw output |\n")
        summary.write("|---|---:|---:|---|\n")
        summary.write(
            f"| SegRDNet | {return_code} | {extract_best_qwk(output)} | [unet.md](unet.md) |\n"
        )

    print(f"\nBaseline summary: {summary_path}")
    return 0 if return_code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
