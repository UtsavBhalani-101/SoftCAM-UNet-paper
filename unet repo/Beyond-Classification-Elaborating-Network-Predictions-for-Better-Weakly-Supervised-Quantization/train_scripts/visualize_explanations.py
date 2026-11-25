"""
Visualize SegRDNet explanation maps for LSE vs average pooling.

Two visualization styles per image:

1. Evidence maps (repo original): forward hooks on fc0..fc3 capture the raw
   per-class evidence maps at each decoder scale BEFORE log-sum-exp / avg
   pooling, rendered as turbo overlays on the input.

2. GradCAM heatmaps (paper style, mirrors demo_code.py): a ClassNet wrapper
   selects one class logit, GradCAM backpropagates through the log-sum-exp
   reduction to the activation feeding it (lse3.check - the Identity between
   the bias shift and log_sum_exp), and the resulting map is rendered with a
   JET colormap at input resolution and saved as raw heatmap PNGs, exactly
   like the paper's demo output.

Usage (Kaggle, from the repo root):
    python train_scripts/visualize_explanations.py \
        --lse-checkpoint /kaggle/input/lse-run/segrdnet/<ts>/best_model.pth \
        --avg-checkpoint /kaggle/input/avg-run/segrdnet/<ts>/best_model.pth \
        --data-dir /kaggle/input/competitions/aptos2019-blindness-detection \
        --num-images 4 \
        --mode both \
        --out-dir /kaggle/working/explanations

GradCAM mode needs: pip install pytorch-grad-cam
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_segrdnet import SegRDNet  # noqa: E402

try:
    from pytorch_grad_cam import GradCAM
    _GRADCAM_ERROR = None
except ImportError as exc:  # message is only fatal when GradCAM mode is used
    GradCAM = None
    _GRADCAM_ERROR = str(exc)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
NUM_CLASSES = 5


class APTOSDataset(Dataset):
    """Same layout as train_aptos.APTOSDataset, kept self-contained."""

    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(os.path.join(self.img_dir, f"{row['id_code']}.png")).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, int(row['diagnosis']) if 'diagnosis' in row else 0


def get_transforms(img_size=512):
    """Validation transform - identical to train_aptos.get_transforms('val')."""
    return T.Compose([
        T.Resize((img_size, img_size), antialias=False),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_state_flexible(checkpoint_path):
    """Accept best_model.pth (raw state_dict) or checkpoint.pt ({'model': ...})."""
    state = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(state, dict):
        for key in ('model', 'state_dict'):
            if key in state and isinstance(state[key], dict) and state[key]:
                state = state[key]
                break
    return {k[len('module.'):] if k.startswith('module.') else k: v
            for k, v in state.items()}


def load_model(pooling, checkpoint, device):
    model = SegRDNet(embedding_dim=1024, num_classes=NUM_CLASSES, pooling=pooling)
    model.load_state_dict(load_state_flexible(checkpoint))
    model.to(device).eval()
    return model


# =============================================================================
# Style 1: per-scale evidence maps via forward hooks
# =============================================================================

def capture_maps(model, x):
    """Run inference and capture the per-class evidence maps fc0..fc3.

    With the SoftCAM Decoder-0 head enabled (mode 'c') fc0 is never called;
    the map is then read from the SoftCAM head instead.
    """
    maps = {}
    hooks = []
    for name in ['fc0', 'fc1', 'fc2', 'fc3']:
        module = getattr(model, name, None)
        if module is None:
            continue

        def hook(_mod, _inp, out, _name=name):
            maps[_name] = out.detach()

        hooks.append(module.register_forward_hook(hook))
    try:
        with torch.no_grad():
            logits = model(x, eval_mode=False)
        if 'fc0' not in maps and getattr(model, 'use_softcam_decoder0', False):
            head = model.softcam_decoder0
            head_module = head.softcam_head if hasattr(head, 'softcam_head') else head
            evidence = head_module.evidence_maps
            if evidence is not None:
                maps['fc0'] = evidence.detach()
        return logits[0], maps
    finally:
        for h in hooks:
            h.remove()


def denormalize(image_tensor):
    img = image_tensor.permute(1, 2, 0).numpy() * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)


def to_overlay(image_tensor, class_map):
    """Overlay a per-class evidence map (turbo colormap) on the image tensor."""
    img = denormalize(image_tensor)

    m = class_map.squeeze(0).cpu().numpy()          # (H, W)
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    m_resized = F.interpolate(
        torch.from_numpy(m)[None, None], size=img.shape[:2], mode='bilinear',
        align_corners=False,
    ).squeeze().numpy()

    heat = plt.get_cmap('turbo')(m_resized)[..., :3]
    return 0.55 * img + 0.45 * heat


def visualize_maps(model, name, image, label, probs, maps, out_dir, img_id):
    """Evidence-map figure: input + fc0..fc3, rows = true / predicted class."""
    pred = int(probs.argmax())

    rows = [(label, 'true class')]
    if pred != label:
        rows.append((pred, 'predicted class'))

    fig, axes = plt.subplots(len(rows), 5, figsize=(20, 4.25 * len(rows)))
    axes = np.atleast_2d(axes)
    if axes.shape[0] == 1:
        axes = axes.reshape(1, -1)

    for row_i, (cls, tag) in enumerate(rows):
        axes[row_i, 0].imshow(denormalize(image))
        axes[row_i, 0].set_title(f'input (id {img_id})\n{tag} {cls}, p={probs[cls]:.3f}', fontsize=9)
        axes[row_i, 0].axis('off')
        for col, dec in enumerate(['fc0', 'fc1', 'fc2', 'fc3'], start=1):
            if dec not in maps:
                axes[row_i, col].axis('off')
                continue
            size = maps[dec].shape[-1]
            blended = to_overlay(image, maps[dec][:, cls])
            axes[row_i, col].imshow(blended)
            axes[row_i, col].set_title(f'{dec} map ({size}x{size})\n{tag}, p={probs[cls]:.3f}', fontsize=9)
            axes[row_i, col].axis('off')

    fig.suptitle(f'{name} pooling | evidence maps | image {img_id} | true grade {label}, predicted {pred}', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{name}_maps_{img_id}_true{label}_pred{pred}.png'), dpi=140)
    plt.close(fig)


# =============================================================================
# Style 2: paper-style GradCAM heatmaps (demo_code.py parity)
# =============================================================================

class ClassNet(nn.Module):
    """Wrapper from the paper's demo: forward returns one class logit."""

    def __init__(self, model, index):
        super().__init__()
        self.model = model
        self.index = index

    def forward(self, x):
        final = self.model(x)
        return final[:, self.index]


class SelectedLogitTarget:
    """Target on the already-selected class logit of ClassNet.

    Equivalent to the demo's ClassNet + BinaryClassifierOutputTarget(1) on raw
    logits, but explicit: GradCAM backpropagates d(logit_cls)/d(activation).
    """

    def __init__(self, cls):
        self.cls = cls

    def __call__(self, model_output):
        return model_output[:, 0]


def gradcam_heatmap(model, x, cls):
    """Paper-style GradCAM at lse3.check for one class; returns HxW in [0, 1]."""
    classifier = ClassNet(model, cls)
    # Layer from the heatmap generation: Identity between the per-class bias
    # shift and the log-so-exp reduction inside the final LSE pooling.
    target_layers = [classifier.model.lse3.check]
    with GradCAM(model=classifier, target_layers=target_layers) as cam:
        grayscale = cam(input_tensor=x, targets=[SelectedLogitTarget(cls)])
    return grayscale[0, :]


def paper_jet_heatmap(gray, size):
    """JET colormap at input resolution, as the paper's demo writes it."""
    cam8 = np.uint8(255 * gray)
    t = torch.from_numpy(cam8.astype(np.float32))[None, None]
    resized = F.interpolate(t, size=(size, size), mode='bicubic', align_corners=False)
    resized = resized.clamp(0, 255).squeeze().numpy().astype(np.uint8)
    return plt.get_cmap('jet')(resized / 255.0)[..., :3]


def visualize_gradcam(model, name, image, label, probs, img_id, out_dir, img_size):
    """Per-class GradCAM panel (paper style) + raw JET heatmap PNGs like demo_code.py."""
    pred = int(probs.argmax())

    fig, axes = plt.subplots(1, NUM_CLASSES + 1, figsize=(4 * (NUM_CLASSES + 1), 4.5))
    axes[0].imshow(denormalize(image))
    axes[0].set_title(f'input (id {img_id})\ntrue {label}, pred {pred}', fontsize=9)
    axes[0].axis('off')

    for cls in range(NUM_CLASSES):
        gray = gradcam_heatmap(model, image.unsqueeze(0), cls)
        rgb_heat = paper_jet_heatmap(gray, img_size)
        # Raw heatmap file with the paper demo's naming (per-class JET map)
        raw_png = os.path.join(out_dir, f'{img_id}_{cls}_{name}.png')
        plt.imsave(raw_png, rgb_heat)

        blended = 0.5 * denormalize(image) + 0.5 * rgb_heat
        tag = ' (pred)' if cls == pred else ''
        axes[cls + 1].imshow(np.clip(blended, 0, 1))
        axes[cls + 1].set_title(f'grade {cls}{tag} | p={probs[cls]:.3f}', fontsize=9)
        axes[cls + 1].axis('off')

    fig.suptitle(f'{name} pooling | GradCAM at lse3.check | image {img_id} | '
                 f'true {label}, predicted {pred}', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{name}_gradcam_{img_id}_true{label}_pred{pred}.png'), dpi=140)
    plt.close(fig)


# =============================================================================
# Driver
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lse-checkpoint', required=True, help='best_model.pth or checkpoint.pt of the LSE run')
    ap.add_argument('--avg-checkpoint', default=None,
                    help='optional AVG-pooling checkpoint to compare against')

    ap.add_argument('--data-dir', default='/kaggle/input/competitions/aptos2019-blindness-detection')
    ap.add_argument('--img-size', type=int, default=512)
    ap.add_argument('--num-images', type=int, default=4)
    ap.add_argument('--mode', default='both', choices=['maps', 'gradcam', 'both'],
                    help='maps: per-scale evidence maps; gradcam: paper-style lse3.check GradCAM')
    ap.add_argument('--out-dir', default='explanations_out')
    args = ap.parse_args()

    if args.mode in ('gradcam', 'both') and GradCAM is None:
        raise SystemExit(f'--mode {args.mode} requires pytorch-grad-cam: pip install pytorch-grad-cam ({_GRADCAM_ERROR})')

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    csv_path = os.path.join(args.data_dir, 'train.csv')
    df = pd.read_csv(csv_path)
    # Spread the sample across grades so collapsed classes are visible too.
    per_grade = args.num_images // NUM_CLASSES or 1
    picks = df.groupby('diagnosis', group_keys=False).apply(lambda g: g.head(per_grade)).head(args.num_images)
    dataset = APTOSDataset(csv_path, os.path.join(args.data_dir, 'train_images'),
                           transform=get_transforms(args.img_size))

    models = {'LSE': load_model('lse', args.lse_checkpoint, device)}
    if args.avg_checkpoint:
        models['AVG'] = load_model('avg', args.avg_checkpoint, device)

    print(f'Visualizing {len(picks)} images across {list(models)} ({args.mode}) -> {args.out_dir}')
    for _, row in picks.iterrows():
        idx = int(df.index[df['id_code'] == row['id_code']][0])
        image, label = dataset[idx]
        img_id = str(row['id_code'])
        for mname, model in models.items():
            logits, maps = capture_maps(model, image.unsqueeze(0))
            probs = torch.softmax(logits, dim=1)[0]
            if args.mode in ('maps', 'both'):
                visualize_maps(model, mname, image, label, probs, maps, args.out_dir, img_id)
            if args.mode in ('gradcam', 'both'):
                visualize_gradcam(model, mname, image, label, probs, img_id, args.out_dir, args.img_size)
            print(f'  [{mname}] saved visualizations for image {img_id} (true {label}, pred {int(probs.argmax())})')

    print('Done.')


if __name__ == '__main__':
    main()
