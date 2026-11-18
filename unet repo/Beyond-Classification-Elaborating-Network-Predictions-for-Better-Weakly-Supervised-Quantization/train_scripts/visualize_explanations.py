"""
Visualize SegRDNet explanation maps for LSE vs average pooling.

Loads one best_model.pth checkpoint per pooling method, runs inference on a
few validation images, captures the per-class evidence maps at all four
decoder scales, and renders side-by-side overlays.

Usage (Kaggle, from the repo root):
    python train_scripts/visualize_explanations.py \
        --lse-checkpoint /kaggle/input/lse-run/best_model.pth \
        --avg-checkpoint /kaggle/input/avg-run/best_model.pth \
        --data-dir /kaggle/input/competitions/aptos2019-blindness-detection \
        --num-images 4 \
        --out-dir /kaggle/working/explanations
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
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_segrdnet import SegRDNet  # noqa: E402

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
    return T.Compose([
        T.Resize((img_size, img_size), antialias=False),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def capture_maps(model, x):
    """Run inference and capture the class evidence maps fc0..fc3."""
    maps = {}
    hooks = []
    for name in ['fc0', 'fc1', 'fc2', 'fc3']:
        module = getattr(model, name)

        def hook(_mod, _inp, out, _name=name):
            maps[_name] = out.detach()

        hooks.append(module.register_forward_hook(hook))
    try:
        with torch.no_grad():
            logits = model(x, eval_mode=False)
        return logits[0], maps
    finally:
        for h in hooks:
            h.remove()


def to_overlay(image_tensor, class_map, title):
    """Overlay a per-class evidence map on the image tensor."""
    img = image_tensor.permute(1, 2, 0).numpy() * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img, 0, 1)

    m = class_map.squeeze(0).cpu().numpy()          # (H, W)
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    m_resized = F.interpolate(
        torch.from_numpy(m)[None, None], size=img.shape[:2], mode='bilinear',
        align_corners=False,
    ).squeeze().numpy()

    cmap = plt.get_cmap('turbo')
    heat = cmap(m_resized)[..., :3]
    blended = 0.55 * img + 0.45 * heat
    return blended, title


def visualize_image(model, name, image, label, img_id, out_dir):
    x = image.unsqueeze(0)
    logits, maps = capture_maps(model, x)
    probs = torch.softmax(logits, dim=1)[0]
    pred = int(probs.argmax())

    fig, axes = plt.subplots(2, 5, figsize=(20, 8.5))
    axes[0, 0].set_ylabel(f'{name}\nQWK-model', fontsize=11)

    img_small = image.permute(1, 2, 0).numpy() * IMAGENET_STD + IMAGENET_MEAN
    img_small = np.clip(img_small, 0, 1)
    axes[0, 0].imshow(img_small)
    axes[0, 0].set_title(f'input (id {img_id})\ntrue {label}, pred {pred}', fontsize=10)
    axes[0, 0].axis('off')

    for col, dec in enumerate(['fc0', 'fc1', 'fc2', 'fc3'], start=1):
        size = maps[dec].shape[-1]
        # Row 0: true class. Row 1: predicted class (blank if identical).
        if label != pred:
            axes[1, 0].axis('off')
            rows = [(0, label, 'true class'), (1, pred, 'predicted class')]
        else:
            axes[1, 0].axis('off')
            rows = [(0, label, 'class'), (1, pred, 'class')]
        for row, cls, tag in rows:
            blended, _ = to_overlay(image, maps[dec][:, cls], '')
            axes[row, col].imshow(blended)
            p = float(probs[cls])
            axes[row, col].set_title(f'{dec} map ({size}x{size})\n{tag}, p={p:.3f}', fontsize=9)
            axes[row, col].axis('off')
            if label == pred and row == 1:
                axes[row, col].clear()
                axes[row, col].axis('off')

    fig.suptitle(f'{name} pooling | image {img_id} | true grade {label}, predicted {pred}', fontsize=13)
    fig.tight_layout()
    path = os.path.join(out_dir, f'{name}_{img_id}_true{label}_pred{pred}.png')
    fig.savefig(path, dpi=140)
    plt.close(fig)


def load_model(pooling, checkpoint, device):
    model = SegRDNet(embedding_dim=1024, num_classes=NUM_CLASSES, pooling=pooling)
    state = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lse-checkpoint', required=True)
    ap.add_argument('--avg-checkpoint', required=True)
    ap.add_argument('--data-dir', default='/kaggle/input/competitions/aptos2019-blindness-detection')
    ap.add_argument('--img-size', type=int, default=512)
    ap.add_argument('--num-images', type=int, default=4)
    ap.add_argument('--out-dir', default='explanations_out')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    csv_path = os.path.join(args.data_dir, 'train.csv')
    df = pd.read_csv(csv_path)
    # Spread the sample across grades so collapsed classes are visible too.
    per_grade = args.num_images // NUM_CLASSES or 1
    picks = df.groupby('diagnosis').head(per_grade).head(args.num_images)
    dataset = APTOSDataset(csv_path, os.path.join(args.data_dir, 'train_images'),
                           transform=get_transforms(args.img_size))

    models = {
        'LSE': load_model('lse', args.lse_checkpoint, device),
        'AVG': load_model('avg', args.avg_checkpoint, device),
    }

    print(f'Visualizing {len(picks)} images across {len(models)} methods -> {args.out_dir}')
    for _, row in picks.iterrows():
        idx = df.index[df['id_code'] == row['id_code']][0]
        image, label = dataset[int(idx)]
        img_id = row['id_code']
        for name, model in models.items():
            visualize_image(model, name, image, label, img_id, args.out_dir)
            print(f'  [{name}] saved maps for image {img_id}')

    print('Done.')


if __name__ == '__main__':
    main()
