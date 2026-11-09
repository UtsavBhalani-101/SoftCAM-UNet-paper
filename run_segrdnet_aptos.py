#!/usr/bin/env python3
"""
SegRDNet on APTOS 2019 - Quick Test Runner
Run directly in Kaggle notebook after cloning the repo.

Usage in Kaggle:
    !git clone https://github.com/chihchiehchen/Beyond-Classification-Elaborating-Network-Predictions-for-Better-Weakly-Supervised-Quantization.git
    %cd Beyond-Classification-Elaborating-Network-Predictions-for-Better-Weakly-Supervised-Quantization
    !python run_segrdnet_aptos.py --epochs 15 --batch-size 16
"""

import os
import sys
import time
import json
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from collections import OrderedDict
from contextlib import suppress
from functools import partial
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision.transforms as T
from PIL import Image

from timm import utils
from timm.data import Mixup, FastCollateMixup
from timm.layers import convert_splitbn_model
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from timm.models import create_model, safe_model_name, resume_checkpoint, model_parameters
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.utils import NativeScaler

from torchmetrics.functional.classification import multiclass_auroc, multiclass_accuracy

# Add current directory to path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.model_segrdnet import SegRDNet


# =============================================================================
# APTOS DATASET
# =============================================================================
class APTOSDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, mode='train'):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.mode = mode
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['id_code']
        label = row['diagnosis'] if 'diagnosis' in row else 0
        
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(img_size=512, mode='train'):
    if mode == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(degrees=30),
            T.RandomAffine(degrees=(-30, 30), translate=(0.1, 0.1), scale=(0.8, 1.2), shear=(-10, 10)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.Resize((img_size, img_size), antialias=False),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size), antialias=False),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


# =============================================================================
# METRICS COMPUTATION
# =============================================================================
def compute_all_metrics(preds, targets, num_classes=5):
    """Compute QWK, Confusion Matrix, Accuracy, AUC (per-class + macro)"""
    preds_np = preds.cpu().numpy()
    targets_np = targets.cpu().numpy()
    pred_labels = np.argmax(preds_np, axis=1)
    
    # QWK (Quadratic Weighted Kappa) - OFFICIAL APTOS METRIC
    qwk = cohen_kappa_score(targets_np, pred_labels, weights='quadratic')
    
    # Confusion Matrix
    cm = confusion_matrix(targets_np, pred_labels, labels=list(range(num_classes)))
    
    # Accuracy
    acc = (pred_labels == targets_np).mean()
    
    # Per-class AUC
    auc_per_class = multiclass_auroc(preds, targets, num_classes=num_classes, average=None).cpu().numpy()
    auc_macro = multiclass_auroc(preds, targets, num_classes=num_classes, average='macro').item()
    
    # Per-class Accuracy
    acc_per_class = multiclass_accuracy(preds, targets, num_classes=num_classes, average=None).cpu().numpy()
    acc_macro = multiclass_accuracy(preds, targets, num_classes=num_classes, average='macro').item()
    
    # Classification Report
    report = classification_report(targets_np, pred_labels, target_names=[f'Class {i}' for i in range(num_classes)], output_dict=True)
    
    return {
        'qwk': qwk,
        'confusion_matrix': cm,
        'accuracy': acc,
        'accuracy_per_class': acc_per_class,
        'accuracy_macro': acc_macro,
        'auc_per_class': auc_per_class,
        'auc_macro': auc_macro,
        'classification_report': report,
        'pred_labels': pred_labels,
        'targets': targets_np
    }


# =============================================================================
# TRAINING & VALIDATION
# =============================================================================
_logger = logging.getLogger('train')

def train_one_epoch(epoch, model, loader, optimizer, loss_fn, args, device, 
                    lr_scheduler=None, amp_autocast=suppress, loss_scaler=None, mixup_fn=None):
    
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    all_predictions = []
    all_targets = []
    losses_m = utils.AverageMeter()

    model.train()
    accum_steps = args.grad_accum_steps
    optimizer.zero_grad()
    
    for batch_idx, (input, target) in enumerate(loader):
        input, target = input.to(device), target.to(device)
        if mixup_fn is not None:
            input, target = mixup_fn(input, target)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        def _forward():
            with amp_autocast():
                output = model(input)
                loss = loss_fn(output, target)
            if accum_steps > 1:
                loss /= accum_steps
            return loss, output

        def _backward(_loss):
            if loss_scaler is not None:
                loss_scaler(_loss, optimizer, clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                           parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                           create_graph=second_order, need_update=True)
            else:
                _loss.backward(create_graph=second_order)
                if args.clip_grad is not None:
                    utils.dispatch_clip_grad(
                        model_parameters(model, exclude_head='agc' in args.clip_mode),
                        value=args.clip_grad, mode=args.clip_mode)
                optimizer.step()

        loss, output = _forward()
        all_predictions += [output]
        all_targets += [target]
        _backward(loss)
        
        losses_m.update(loss.item() * accum_steps, input.size(0))
        optimizer.zero_grad()

        if batch_idx % args.log_interval == 0:
            lrl = [pg['lr'] for pg in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            _logger.info(f'Train: {epoch} [{batch_idx:>4d}/{len(loader)}]  '
                       f'Loss: {losses_m.val:.4f} ({losses_m.avg:.4f})  LR: {lr:.2e}')

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=epoch * len(loader) + batch_idx, metric=losses_m.avg)

    p_test = torch.cat(all_predictions, dim=0)
    y_test = torch.cat(all_targets, dim=0)
    
    metrics = compute_all_metrics(p_test, y_test, args.num_classes)
    metrics['loss'] = losses_m.avg
    
    print(f"  Train: Loss={metrics['loss']:.4f}, QWK={metrics['qwk']:.4f}, Acc={metrics['accuracy']:.4f}, AUC={metrics['auc_macro']:.4f}")
    print(f"  AUC per class: {metrics['auc_per_class']}")
    
    return metrics


def validate(model, loader, loss_fn, args, device, amp_autocast=suppress):
    losses_m = utils.AverageMeter()
    all_predictions = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            input = input.to(device)
            target = target.to(device)
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)
                if isinstance(output, (tuple, list)):
                    output = output[0]
                loss = loss_fn(output, target)

            all_predictions += [output]
            all_targets += [target]
            losses_m.update(loss.item(), input.size(0))

    p_test = torch.cat(all_predictions, dim=0)
    y_test = torch.cat(all_targets, dim=0)
    
    metrics = compute_all_metrics(p_test, y_test, args.num_classes)
    metrics['loss'] = losses_m.avg
    
    print(f"  Valid: Loss={metrics['loss']:.4f}, QWK={metrics['qwk']:.4f}, Acc={metrics['accuracy']:.4f}, AUC={metrics['auc_macro']:.4f}")
    print(f"  AUC per class: {metrics['auc_per_class']}")
    print(f"  Confusion Matrix:\n{metrics['confusion_matrix']}")
    
    return metrics


# =============================================================================
# SUBMISSION GENERATION
# =============================================================================
def generate_submission(model, output_dir, args, device, transform):
    model.eval()
    test_csv_path = os.path.join(args.data_dir, args.test_csv)
    test_img_dir = os.path.join(args.data_dir, args.test_img_dir)
    test_df = pd.read_csv(test_csv_path)
    
    test_dataset = APTOSDataset(test_csv_path, test_img_dir, transform=transform, mode='val')
    test_loader = DataLoader(test_dataset, batch_size=args.val_batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)
    
    predictions = []
    with torch.no_grad():
        for input, _ in test_loader:
            input = input.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp):
                output = model(input)
                if isinstance(output, (tuple, list)):
                    output = output[0]
                probs = torch.softmax(output, dim=1)
                preds = torch.argmax(probs, dim=1)
                predictions.extend(preds.cpu().numpy())
    
    submission = pd.DataFrame({'id_code': test_df['id_code'], 'diagnosis': predictions})
    submission_path = os.path.join(output_dir, 'submission.csv')
    submission.to_csv(submission_path, index=False)
    print(f'Submission saved: {submission_path}')
    print(submission['diagnosis'].value_counts().sort_index())
    
    return submission


# =============================================================================
# MAIN
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='SegRDNet APTOS Quick Test')
    
    # Data
    parser.add_argument('--data-dir', default='/kaggle/input/competitions/aptos2019-blindness-detection', type=str)
    parser.add_argument('--train-csv', default='train.csv', type=str)
    parser.add_argument('--test-csv', default='test.csv', type=str)
    parser.add_argument('--train-img-dir', default='train_images', type=str)
    parser.add_argument('--test-img-dir', default='test_images', type=str)
    
    # Model
    parser.add_argument('--num-classes', default=5, type=int)
    
    # Training
    parser.add_argument('--epochs', default=15, type=int, help='Number of epochs (10-20 for quick test)')
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--val-batch-size', default=32, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--img-size', default=512, type=int)
    
    # Optimizer
    parser.add_argument('--opt', default='adamw', type=str)
    parser.add_argument('--lr', default=None, type=float, help='Learning rate (auto if None)')
    parser.add_argument('--lr-base', default=1e-3, type=float)
    parser.add_argument('--lr-base-size', default=256, type=int)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--sched', default='cosine', type=str)
    parser.add_argument('--warmup-epochs', default=3, type=int)
    parser.add_argument('--min-lr', default=1e-6, type=float)
    
    # Regularization
    parser.add_argument('--drop', default=0.0, type=float)
    parser.add_argument('--drop-path', default=0.1, type=float)
    parser.add_argument('--smoothing', default=0.1, type=float)
    parser.add_argument('--mixup', default=0.5, type=float)
    parser.add_argument('--cutmix', default=1.0, type=float)
    parser.add_argument('--mixup-prob', default=1.0, type=float)
    parser.add_argument('--mixup-switch-prob', default=0.5, type=float)
    parser.add_argument('--mixup-mode', default='batch', type=str)
    parser.add_argument('--mixup-off-epoch', default=0, type=int)
    
    # System
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--amp', action='store_true', default=True)
    parser.add_argument('--no-amp', dest='amp', action='store_false')
    parser.add_argument('--amp-dtype', default='float16', type=str)
    parser.add_argument('--output', default='/kaggle/working/segrdnet_aptos', type=str)
    parser.add_argument('--experiment', default='segrdnet_aptos', type=str)
    parser.add_argument('--eval-metric', default='auc_mean', type=str)
    parser.add_argument('--log-interval', default=20, type=int)
    parser.add_argument('--checkpoint-hist', default=3, type=int)
    
    # Distributed (not used)
    parser.add_argument('--local-rank', default=0, type=int)
    
    # Other
    parser.add_argument('--grad-accum-steps', default=1, type=int)
    parser.add_argument('--channels-last', action='store_true', default=False)
    parser.add_argument('--clip-grad', default=None, type=float)
    parser.add_argument('--clip-mode', default='norm', type=str)
    parser.add_argument('--prefetcher', action='store_true', default=True)
    parser.add_argument('--no-prefetcher', dest='prefetcher', action='store_false')
    
    return parser.parse_args()


def main():
    args = parse_args()
    utils.setup_default_logging()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    args.rank = 0
    args.world_size = 1
    args.distributed = False
    
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # AMP setup
    use_amp = args.amp and hasattr(torch.cuda.amp, 'autocast')
    amp_dtype = torch.float16 if args.amp_dtype == 'float16' else torch.bfloat16
    amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype) if use_amp else suppress
    loss_scaler = NativeScaler() if use_amp else None
    print(f"AMP: {'Enabled' if use_amp else 'Disabled'}")
    
    utils.random_seed(args.seed, args.rank)
    
    # Create model
    print(f'Creating SegRDNet (num_classes={args.num_classes})...')
    model = SegRDNet(num_classes=args.num_classes)
    model.to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Data config for timm
    data_config = resolve_data_config(vars(args), model=model, verbose=False)
    
    # DataLoaders
    train_transform = get_transforms(args.img_size, 'train')
    val_transform = get_transforms(args.img_size, 'val')
    
    train_csv_path = os.path.join(args.data_dir, args.train_csv)
    train_img_dir = os.path.join(args.data_dir, args.train_img_dir)
    
    full_dataset = APTOSDataset(train_csv_path, train_img_dir, transform=train_transform, mode='train')
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transform
    val_dataset.dataset.mode = 'val'
    
    print(f'Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}')
    
    # Mixup
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0
    if mixup_active:
        mixup_args = dict(mixup_alpha=args.mixup, cutmix_alpha=args.cutmix,
                         cutmix_minmax=args.cutmix_minmax if hasattr(args, 'cutmix_minmax') else None,
                         prob=args.mixup_prob, switch_prob=args.mixup_switch_prob,
                         mode=args.mixup_mode, label_smoothing=args.smoothing,
                         num_classes=args.num_classes)
        if args.prefetcher:
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)
    else:
        collate_fn = None
    
    loader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, drop_last=True, collate_fn=collate_fn, pin_memory=True)
    loader_eval = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)
    
    # Loss
    if mixup_active:
        train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    train_loss_fn = train_loss_fn.to(device)
    validate_loss_fn = nn.CrossEntropyLoss().to(device)
    
    # Optimizer & Scheduler
    if not args.lr:
        global_batch_size = args.batch_size * args.world_size * args.grad_accum_steps
        batch_ratio = global_batch_size / args.lr_base_size
        args.lr_base_scale = 'sqrt' if any(o in args.opt.lower() for o in ('ada', 'lamb')) else 'linear'
        if args.lr_base_scale == 'sqrt':
            batch_ratio = batch_ratio ** 0.5
        args.lr = args.lr_base * batch_ratio
    print(f'Learning rate: {args.lr:.2e}')
    
    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args), **getattr(args, 'opt_kwargs', {}))
    
    updates_per_epoch = (len(loader_train) + args.grad_accum_steps - 1) // args.grad_accum_steps
    lr_scheduler, _ = create_scheduler_v2(optimizer, **scheduler_kwargs(args), updates_per_epoch=updates_per_epoch)
    
    # Output dir
    exp_name = f"{args.experiment}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output_dir = utils.get_outdir(args.output, exp_name)
    print(f"Output dir: {output_dir}")
    
    # Training history
    history = {'train': [], 'val': []}
    best_qwk = 0
    best_auc = 0
    best_epoch = 0
    
    print(f'\n=== Starting Training for {args.epochs} epochs ===')
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_one_epoch(epoch, model, loader_train, optimizer, train_loss_fn,
                                       args, device, lr_scheduler, amp_autocast, loss_scaler, mixup_fn)
        
        # Validate
        val_metrics = validate(model, loader_eval, validate_loss_fn, args, device, amp_autocast)
        
        # History
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)
        
        # Save best model (by QWK - official APTOS metric)
        if val_metrics['qwk'] > best_qwk:
            best_qwk = val_metrics['qwk']
            best_auc = val_metrics['auc_macro']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'qwk': best_qwk,
                'auc': best_auc,
                'args': args
            }, os.path.join(output_dir, 'best_model.pth'))
            print(f"  >>> Best model saved (QWK: {best_qwk:.4f}, AUC: {best_auc:.4f})")
        
        if lr_scheduler is not None:
            lr_scheduler.step(epoch + 1, val_metrics[args.eval_metric])
        
        epoch_time = time.time() - epoch_start
        print(f"  Epoch time: {epoch_time:.1f}s")
    
    total_time = time.time() - start_time
    print(f'\n=== Training Complete ===')
    print(f'Total time: {total_time/60:.1f} min')
    print(f'Best QWK: {best_qwk:.4f} at epoch {best_epoch}')
    print(f'Best AUC: {best_auc:.4f}')
    
    # Save history
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        return obj
    
    with open(os.path.join(output_dir, 'history.json'), 'w') as f:
        json.dump(convert_for_json(history), f, indent=2)
    
    # Final evaluation with best model
    print('\n=== Final Evaluation (Best Model) ===')
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pth'), map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    final_metrics = validate(model, loader_eval, validate_loss_fn, args, device, amp_autocast)
    
    print(f"\nFINAL RESULTS:")
    print(f"  QWK: {final_metrics['qwk']:.4f}")
    print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"  Macro AUC: {final_metrics['auc_macro']:.4f}")
    print(f"  Per-class AUC: {final_metrics['auc_per_class']}")
    print(f"  Confusion Matrix:")
    print(final_metrics['confusion_matrix'])
    
    # Generate submission
    generate_submission(model, output_dir, args, device, val_transform)
    
    return model, final_metrics, output_dir


if __name__ == '__main__':
    main()