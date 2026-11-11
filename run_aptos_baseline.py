#!/usr/bin/env python3
"""Run SoftCAM and SegRDNet APTOS baselines and archive raw console output.

The SoftCAM baseline uses its APTOS trainer, which does not add the SoftCAM
activation-map L1/L2 penalty. The SegRDNet baseline leaves
--use-softcam-decoder0 disabled; its APTOS trainer optimizes final-decoder
cross-entropy only and does not add the returned multi-scale consistency loss.

Run from the project root (or pass --project-root) in Kaggle. Each run's raw
stdout/stderr is streamed to the notebook and saved as a Markdown log.
"""

from __future__ import annotations

import argparse
import math
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
    raise SystemExit("PyTorch is required to determine the available GPU count.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--data-dir",
        default="/kaggle/input/competitions/aptos2019-blindness-detection",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32, help="Target effective global batch size")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--softcam-network", default="resnet50")
    parser.add_argument("--output", default="/kaggle/working/aptos_baseline")
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


def find_scripts(root: Path) -> tuple[Path, Path]:
    softcam_candidates = [
        root / "softcam repo" / "SoftCAM" / "train_aptos.py",
        root / "softcam_repo" / "SoftCAM" / "train_aptos.py",
    ]
    unet_project = "Beyond-Classification-Elaborating-Network-Predictions-for-Better-Weakly-Supervised-Quantization"
    unet_candidates = [
        root / repo_name / unet_project / "train_scripts" / "train_aptos.py"
        for repo_name in ("unet repo", "unet_repo")
    ]
    softcam_script = next((path for path in softcam_candidates if path.is_file()), None)
    unet_script = next((path for path in unet_candidates if path.is_file()), None)
    if softcam_script is None:
        raise SystemExit("Could not locate SoftCAM train_aptos.py under the project root.")
    if unet_script is None:
        raise SystemExit("Could not locate UNet train_scripts/train_aptos.py under the project root.")
    return softcam_script, unet_script


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    softcam_script, unet_script = find_scripts(root)
    softcam_dir = softcam_script.parent
    unet_dir = unet_script.parent.parent

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    softcam_batch_size = max(1, math.ceil(args.batch_size / max(1, gpu_count)))
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    archive_dir = Path(args.output) / "baseline_raw_logs" / run_id
    softcam_output = str(Path(args.output) / "softcam")
    unet_output = str(Path(args.output) / "unet")

    common = [
        "--data-dir", args.data_dir,
        "--epochs", str(args.epochs),
        "--workers", str(args.workers),
        "--img-size", str(args.img_size),
        "--seed", str(args.seed),
    ]
    softcam_command = [
        sys.executable, str(softcam_script),
        *common,
        "--network", args.softcam_network,
        "--batch-size", str(softcam_batch_size),
        "--output", softcam_output,
    ]
    unet_command = [
        sys.executable, str(unet_script),
        *common,
        "--batch-size", str(args.batch_size),
        "--output", unet_output,
    ]

    print("Baseline settings:")
    print("  SoftCAM activation-map L1/L2 penalty: disabled by its APTOS trainer")
    print("  SegRDNet multi-scale consistency loss: not added by its APTOS trainer")
    print(f"  Target effective batch size: {args.batch_size}")
    print(f"  GPU count: {gpu_count}; SoftCAM per-GPU batch argument: {softcam_batch_size}")
    print(f"  Raw Markdown logs: {archive_dir}")

    results: list[tuple[str, int, str, Path]] = []
    for label, command, cwd, filename in (
        ("SoftCAM", softcam_command, softcam_dir, "softcam.md"),
        ("SegRDNet without consistency loss", unet_command, unet_dir, "unet.md"),
    ):
        log_path = archive_dir / filename
        print(f"\n{'=' * 72}\nStarting {label}\n{'=' * 72}")
        return_code, output = run_and_archive(label, command, cwd, log_path)
        results.append((label, return_code, extract_best_qwk(output), log_path))

    summary_path = archive_dir / "baseline_summary.md"
    with summary_path.open("w", encoding="utf-8") as summary:
        summary.write("# APTOS baseline comparison\n\n")
        summary.write(f"- Seed: {args.seed}\n- Epochs: {args.epochs}\n")
        summary.write(f"- Target effective batch size: {args.batch_size}\n")
        summary.write("- SoftCAM map regularization: disabled\n")
        summary.write("- SegRDNet consistency loss: disabled\n\n")
        summary.write("| Model | Exit code | Best validation QWK | Raw output |\n")
        summary.write("|---|---:|---:|---|\n")
        for label, return_code, qwk, log_path in results:
            summary.write(f"| {label} | {return_code} | {qwk} | [{log_path.name}]({log_path.name}) |\n")

    print(f"\nBaseline summary: {summary_path}")
    return 0 if all(return_code == 0 for _, return_code, _, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
