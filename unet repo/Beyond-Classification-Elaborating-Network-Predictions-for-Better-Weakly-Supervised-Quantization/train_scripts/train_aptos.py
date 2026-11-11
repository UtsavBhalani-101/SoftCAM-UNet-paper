#!/usr/bin/env python3
"""
Chen's Unet (SegRDNet) APTOS 2019 Training Script
Single-file entry point for 5-class DR classification on APTOS dataset.
Uses existing SegRDNet model code from models/model_segrdnet.py
"""
import os

# Suppress CUDA driver warnings BEFORE importing torch
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import time
import random
import argparse
import logging
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
from torch.utils.tensorboard import SummaryWriter

import torchvision.transforms as T
from PIL import Image

from torchmetrics.functional.classification import multiclass_auroc, multiclass_accuracy, multiclass_cohen_kappa

# =============================================================================
# IMPORT EXISTING CHEN'S UNET MODEL CODE
# =============================================================================
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
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
# METRICS
# =============================================================================
def compute_all_metrics(preds, targets, num_classes=5):
    """Compute QWK, Confusion Matrix, Accuracy, AUC (per-class + macro)"""
    preds_np = preds.cpu().numpy()
    targets_np = targets.cpu().numpy()
    pred_labels = np.argmax(preds_np, axis=1)
    
    # QWK - Official APTOS metric
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
    
    # Kappa
    kappa = multiclass_cohen_kappa(preds, targets, num_classes=num_classes).item()
    
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
        'kappa': kappa,
        'classification_report': report,
        'pred_labels': pred_labels,
        'targets': targets_np
    }


# =============================================================================
# TRAINING & VALIDATION
# =============================================================================
def train_one_epoch(epoch, model, loader, optimizer, loss_fn, cfg, device, 
                    lr_scheduler=None, amp_autocast=suppress, loss_scaler=None):
    
    all_predictions = []
    all_targets = []
    losses_m = 0.0
    num_batches = 0

    model.train()
    optimizer.zero_grad()
    
    grad_accum_steps = getattr(cfg, 'grad_accum_steps', 1)
    
    # Check if model has SoftCAM regularization
    has_softcam = hasattr(model, 'get_softcam_regularization')
    
    for batch_idx, (input, target) in enumerate(loader):
        input = input.to(device)
        target = target.to(device)
        
        # Mixed precision forward
        with amp_autocast():
            if cfg.use_softcam_decoder0:
                output, _, scale_logits = model(input, eval_mode=False)
                # SegRDNet returns scale logits as (decoder3, decoder2,
                # decoder1, decoder0). Explicitly supervise the new decoder0
                # SoftCAM head so it learns class evidence, not just small maps.
                softcam_aux_loss = loss_fn(scale_logits[-1], target)
                loss = loss_fn(output, target) + cfg.softcam_aux_weight * softcam_aux_loss
            else:
                output = model(input)
                loss = loss_fn(output, target)
            
            # Add SoftCAM regularization loss (L1/L2 on evidence maps)
            if has_softcam:
                softcam_reg = model.get_softcam_regularization()
                if softcam_reg != 0.0:
                    loss = loss + softcam_reg
        
        # Scale loss for gradient accumulation
        loss = loss / grad_accum_steps
        
        # Backward
        if loss_scaler is not None:
            loss_scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Step optimizer only every grad_accum_steps
        if (batch_idx + 1) % grad_accum_steps == 0:
            if loss_scaler is not None:
                if cfg.grad_clip:
                    loss_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                loss_scaler.step(optimizer)
                loss_scaler.update()
            else:
                if cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
            optimizer.zero_grad()
        
        # Use unscaled loss for logging
        losses_m += loss.item() * grad_accum_steps
        num_batches += 1
        all_predictions.append(output.detach())
        all_targets.append(target.detach())
        
        if batch_idx % cfg.log_interval == 0:
            print(f'Epoch {epoch} [{batch_idx}/{len(loader)}] Loss: {loss.item() * grad_accum_steps:.4f}')
    
    # Handle remaining gradients
    if (len(loader) % grad_accum_steps) != 0:
        if loss_scaler is not None:
            if cfg.grad_clip:
                loss_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            loss_scaler.step(optimizer)
            loss_scaler.update()
        else:
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
        optimizer.zero_grad()
    
    avg_loss = losses_m / max(1, num_batches)
    p_test = torch.cat(all_predictions, dim=0)
    y_test = torch.cat(all_targets, dim=0)
    
    metrics = compute_all_metrics(p_test, y_test, cfg.num_classes)
    metrics['loss'] = avg_loss
    
    print(f"  Train: Loss={metrics['loss']:.4f}, QWK={metrics['qwk']:.4f}, Acc={metrics['accuracy']:.4f}, AUC={metrics['auc_macro']:.4f}")
    print(f"  AUC per class: {metrics['auc_per_class']}")
    
    if lr_scheduler is not None and not isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        lr_scheduler.step()
    
    return metrics


def validate(model, loader, loss_fn, cfg, device, amp_autocast=suppress):
    losses_m = 0.0
    all_predictions = []
    all_targets = []
    num_batches = 0

    model.eval()
    with torch.no_grad():
        for input, target in loader:
            input = input.to(device)
            target = target.to(device)
            
            with amp_autocast():
                output = model(input)
                loss = loss_fn(output, target)
            
            losses_m += loss.item()
            num_batches += 1
            all_predictions.append(output)
            all_targets.append(target)
    
    avg_loss = losses_m / max(1, num_batches)
    p_test = torch.cat(all_predictions, dim=0)
    y_test = torch.cat(all_targets, dim=0)
    
    metrics = compute_all_metrics(p_test, y_test, cfg.num_classes)
    metrics['loss'] = avg_loss
    
    print(f"  Valid: Loss={metrics['loss']:.4f}, QWK={metrics['qwk']:.4f}, Acc={metrics['accuracy']:.4f}, AUC={metrics['auc_macro']:.4f}, Kappa={metrics['kappa']:.4f}")
    print(f"  AUC per class: {metrics['auc_per_class']}")
    print(f"  Acc per class: {metrics['accuracy_per_class']}")
    print(f"  Confusion Matrix:\n{metrics['confusion_matrix']}")
    
    return metrics


# =============================================================================
# SUBMISSION GENERATION
# =============================================================================
def generate_submission(cfg, model, save_path, val_transform):
    model.eval()
    
    test_csv_path = os.path.join(cfg.data_dir, cfg.test_csv)
    test_img_dir = os.path.join(cfg.data_dir, cfg.test_img_dir)
    test_df = pd.read_csv(test_csv_path)
    
    test_dataset = APTOSDataset(test_csv_path, test_img_dir, transform=val_transform, mode='val')
    test_loader = DataLoader(test_dataset, batch_size=cfg.val_batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)
    
    predictions = []
    with torch.no_grad():
        for input, _ in test_loader:
            input = input.to(cfg.device)
            with torch.autocast(device_type=cfg.device.type, dtype=torch.float16, enabled=cfg.amp):
                output = model(input)
            probs = torch.softmax(output, dim=1)
            preds = torch.argmax(probs, dim=1)
            predictions.extend(preds.cpu().numpy())
    
    submission = pd.DataFrame({'id_code': test_df['id_code'], 'diagnosis': predictions})
    submission_path = os.path.join(save_path, 'submission.csv')
    submission.to_csv(submission_path, index=False)
    print(f'Submission saved: {submission_path}')
    print(submission['diagnosis'].value_counts().sort_index())
    return submission


# =============================================================================
# CONFIG CLASS
# =============================================================================
class Config:
    def __init__(self, args):
        # Device detection - simple check
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        # Data paths
        self.data_dir = args.data_dir
        self.train_csv = args.train_csv
        self.test_csv = args.test_csv
        self.train_img_dir = args.train_img_dir
        self.test_img_dir = args.test_img_dir
        self.img_size = args.img_size
        
        # Training - single GPU for SegRDNet (hooks incompatible with DataParallel)
        # Reduced batch size for 512x512 on T4 16GB
        self.batch_size = min(args.batch_size, 8)
        self.val_batch_size = min(args.val_batch_size, 16)
        
        # Gradient accumulation to maintain effective batch size
        self.grad_accum_steps = max(1, args.batch_size // self.batch_size)
        self.num_workers = args.workers
        self.persistent_workers = True if args.workers > 0 else False
        self.num_classes = args.num_classes
        
        # Model (fixed: SegRDNet with RDNet Tiny backbone)
        self.epochs = args.epochs
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.optimizer = args.optimizer
        self.scheduler = args.scheduler
        self.label_smoothing = args.label_smoothing
        self.grad_clip = args.grad_clip
        self.log_interval = args.log_interval
        
        # Option A: SoftCAM Decoder 0
        self.use_softcam_decoder0 = args.use_softcam_decoder0
        self.softcam_lambda_l1 = args.softcam_lambda_l1
        self.softcam_lambda_l2 = args.softcam_lambda_l2
        self.softcam_aux_weight = args.softcam_aux_weight
        
        # AMP
        self.amp = args.amp
        self.amp_dtype = args.amp_dtype
        
        # Output
        self.save_path = args.output


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_save_paths(cfg):
    save_path = os.path.join(cfg.save_path, 'segrdnet', datetime.now().strftime("%Y%m%d-%H%M%S"))
    return save_path


# =============================================================================
# MAIN
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Chen's Unet (SegRDNet) APTOS 2019 Training")
    
    # Data paths (Kaggle default)
    parser.add_argument('--data-dir', default='/kaggle/input/competitions/aptos2019-blindness-detection', type=str)
    parser.add_argument('--train-csv', default='train.csv', type=str)
    parser.add_argument('--test-csv', default='test.csv', type=str)
    parser.add_argument('--train-img-dir', default='train_images', type=str)
    parser.add_argument('--test-img-dir', default='test_images', type=str)
    
    # Model
    parser.add_argument('--num-classes', default=5, type=int)
    
    # Training - USER SPECIFIED
    parser.add_argument('--epochs', default=20, type=int, help='20 epochs')
    parser.add_argument('--batch-size', default=32, type=int, help='32 batch size')
    parser.add_argument('--val-batch-size', default=32, type=int)
    parser.add_argument('--workers', default=4, type=int, help='4 persistent workers')
    parser.add_argument('--img-size', default=512, type=int)
    
    # Optimizer
    parser.add_argument('--optimizer', default='adamw', type=str, choices=['adamw', 'sgd'])
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--scheduler', default='cosine', type=str, choices=['cosine', 'step', 'plateau'])
    
    # Loss
    parser.add_argument('--label-smoothing', default=0.1, type=float)
    
    # Regularization
    parser.add_argument('--grad-clip', default=1.0, type=float)
    
    # Option A: SoftCAM Decoder 0
    parser.add_argument('--use-softcam-decoder0', action='store_true', default=False, help='Enable SoftCAM evidence head for Decoder 0')
    parser.add_argument('--softcam-lambda-l1', default=0.0, type=float, help='L1 regularization weight for SoftCAM evidence maps')
    parser.add_argument('--softcam-lambda-l2', default=0.0, type=float, help='L2 regularization weight for SoftCAM evidence maps')
    parser.add_argument('--softcam-aux-weight', default=0.25, type=float, help='Weight for Decoder 0 SoftCAM classification loss')
    
    # System
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--output', default='/kaggle/working/segrdnet_aptos', type=str)
    parser.add_argument('--log-interval', default=20, type=int)
    parser.add_argument('--amp', action='store_true', default=True)
    parser.add_argument('--no-amp', dest='amp', action='store_false')
    parser.add_argument('--amp-dtype', default='float16', type=str)
    
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config(args)
    
    # Print config - NO driver detection, just is_available
    print("="*60)
    print("CHEN'S UNET (SegRDNet) APTOS TRAINING CONFIG")
    print("="*60)
    print(f"Device: {cfg.device}")
    print(f"GPUs available: {cfg.num_gpus}")
    if cfg.num_gpus > 0:
        print(f"  Using single GPU (cuda:0) - SegRDNet hooks incompatible with DataParallel")
    print(f"Epochs: {cfg.epochs}")
    print(f"Batch Size: {cfg.batch_size} (effective: {cfg.batch_size * getattr(cfg, 'grad_accum_steps', 1)} with grad_accum={getattr(cfg, 'grad_accum_steps', 1)})")
    print(f"Workers: {cfg.num_workers} (persistent: {cfg.persistent_workers})")
    print(f"Loss: CrossEntropy (label_smoothing={cfg.label_smoothing})")
    
    # Option A: SoftCAM Decoder 0
    if cfg.use_softcam_decoder0:
        print(f"SoftCAM Decoder 0: ENABLED")
        print(f"  L1 lambda: {cfg.softcam_lambda_l1}")
        print(f"  L2 lambda: {cfg.softcam_lambda_l2}")
    else:
        print(f"SoftCAM Decoder 0: DISABLED (baseline)")
    
    print(f"Output: {cfg.save_path}")
    print("="*60)
    
    set_random_seed(args.seed)
    
    save_path = load_save_paths(cfg)
    os.makedirs(save_path, exist_ok=True)
    print(f'Save path: {save_path}')
    
    logger = SummaryWriter(save_path)
    
    # AMP setup
    use_amp = cfg.amp and hasattr(torch.cuda.amp, 'autocast')
    amp_dtype = torch.float16 if cfg.amp_dtype == 'float16' else torch.bfloat16
    amp_autocast = partial(torch.autocast, device_type=cfg.device.type, dtype=amp_dtype) if use_amp else suppress
    loss_scaler = torch.amp.GradScaler() if use_amp else None
    print(f"AMP: {'Enabled' if use_amp else 'Disabled'}")
    
    # Data
    train_transform = get_transforms(cfg.img_size, 'train')
    val_transform = get_transforms(cfg.img_size, 'val')
    
    train_csv_path = os.path.join(cfg.data_dir, cfg.train_csv)
    train_img_dir = os.path.join(cfg.data_dir, cfg.train_img_dir)
    
    full_dataset = APTOSDataset(train_csv_path, train_img_dir, transform=train_transform, mode='train')
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transform
    val_dataset.dataset.mode = 'val'
    
    print(f'Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True,
        num_workers=cfg.num_workers, 
        pin_memory=True, 
        drop_last=True,
        persistent_workers=cfg.persistent_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.val_batch_size, 
        shuffle=False,
        num_workers=cfg.num_workers, 
        pin_memory=True,
        persistent_workers=cfg.persistent_workers
    )
    
    # Model - use Average pooling instead of LSE
    model = SegRDNet(
        num_classes=cfg.num_classes, 
        pooling='avg',
        use_softcam_decoder0=cfg.use_softcam_decoder0,
        softcam_lambda_l1=cfg.softcam_lambda_l1,
        softcam_lambda_l2=cfg.softcam_lambda_l2
    )
    
    model.to(cfg.device)
    
    # NOTE: SegRDNet uses forward hooks for multi-scale feature extraction
    # which are incompatible with nn.DataParallel.
    # Use single GPU (will use GPU 0). For multi-GPU, consider DDP instead.
    if cfg.num_gpus > 1:
        print(f"Multiple GPUs detected ({cfg.num_gpus}), but SegRDNet uses forward hooks")
        print("  incompatible with DataParallel. Using single GPU (cuda:0).")
        print("  For multi-GPU training, use DistributedDataParallel (DDP) instead.")
    
    print(f'Model: SegRDNet, Parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Loss
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing).to(cfg.device)
    
    # Optimizer
    if cfg.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f'Unknown optimizer: {cfg.optimizer}')
    
    # Scheduler
    if cfg.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.01)
    elif cfg.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.epochs//3, gamma=0.1)
    elif cfg.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    else:
        scheduler = None
    
    # Training
    print('\nStarting training...')
    best_metric = 0.0
    best_epoch = 0
    
    for epoch in range(cfg.epochs):
        epoch_start = time.time()
        
        train_metrics = train_one_epoch(epoch, model, train_loader, optimizer, loss_fn, 
                                       cfg, cfg.device, scheduler, amp_autocast, loss_scaler)
        val_metrics = validate(model, val_loader, loss_fn, cfg, cfg.device, amp_autocast)
        
        # Logging
        logger.add_scalar('Loss/train', train_metrics['loss'], epoch)
        logger.add_scalar('Loss/val', val_metrics['loss'], epoch)
        logger.add_scalar('AUC/train', train_metrics['auc_macro'], epoch)
        logger.add_scalar('AUC/val', val_metrics['auc_macro'], epoch)
        logger.add_scalar('Acc/train', train_metrics['accuracy'], epoch)
        logger.add_scalar('Acc/val', val_metrics['accuracy'], epoch)
        logger.add_scalar('QWK/val', val_metrics['qwk'], epoch)
        logger.add_scalar('Kappa/val', val_metrics['kappa'], epoch)
        
        # Save best model (by QWK)
        metric = val_metrics['qwk']
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, os.path.join(save_path, 'best_model.pth'))
            print(f'  -> Best model saved (QWK: {best_metric:.4f})')
        
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metric)
    
    print(f'\n{"="*60}')
    print(f'Best validation QWK: {best_metric:.4f} at epoch {best_epoch}')
    print(f'{"="*60}')
    
    # Final evaluation with best model
    print('\n=== Final Evaluation (Best Model) ===')
    best_state = torch.load(os.path.join(save_path, 'best_model.pth'), map_location=cfg.device)
    model.load_state_dict(best_state)
    final_metrics = validate(model, val_loader, loss_fn, cfg, cfg.device, amp_autocast)
    print(f"FINAL: QWK={final_metrics['qwk']:.4f}, Acc={final_metrics['accuracy']:.4f}, AUC={final_metrics['auc_macro']:.4f}")
    print(f"Confusion Matrix:\n{final_metrics['confusion_matrix']}")
    
    # Generate submission
    print('\nGenerating submission...')
    generate_submission(cfg, model, save_path, val_transform)
    
    logger.close()
    print('\nDone!')


if __name__ == '__main__':
    main()
