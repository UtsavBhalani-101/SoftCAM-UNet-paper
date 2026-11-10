#!/usr/bin/env python3
"""
SoftCAM APTOS 2019 Training Script
Single-file entry point for 5-class DR classification on APTOS dataset.
Uses existing SoftCAM model code (FullyConvModel, ConvClassifier, losses).
"""

# Suppress CUDA driver warnings BEFORE importing torch
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow/absl logs

import os
import sys
import time
import random
import argparse
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from collections import OrderedDict
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import torchvision.transforms as T
from torchvision import models
from PIL import Image

from torchmetrics.functional.classification import multiclass_auroc, multiclass_accuracy, multiclass_cohen_kappa

# =============================================================================
# IMPORT EXISTING SOFTCAM MODEL CODE
# =============================================================================
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.models import FullyConvModel, ConvClassifier
from modules.loss import FocalLoss, KappaLoss
from utils.func import one_hot


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
        
        return {'image': image, 'target': torch.tensor(label, dtype=torch.long)}


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
# MODEL GENERATION (adapted from modules/builder.py)
# =============================================================================
def generate_model(cfg):
    network = cfg.network
    num_classes = cfg.num_classes
    pretrained = cfg.pretrained
    
    # Load backbone from torchvision
    if network == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Identity()
        num_ftrs = 2048
    elif network == 'resnet101':
        model = models.resnet101(pretrained=pretrained)
        model.fc = nn.Identity()
        num_ftrs = 2048
    elif network == 'resnet152':
        model = models.resnet152(pretrained=pretrained)
        model.fc = nn.Identity()
        num_ftrs = 2048
    elif network == 'densenet121':
        model = models.densenet121(pretrained=pretrained)
        model.classifier = nn.Identity()
        num_ftrs = 1024
    elif network == 'densenet169':
        model = models.densenet169(pretrained=pretrained)
        model.classifier = nn.Identity()
        num_ftrs = 1664
    elif network == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
        model.classifier = nn.Identity()
        num_ftrs = 1280
    elif network == 'efficientnet_b3':
        model = models.efficientnet_b3(pretrained=pretrained)
        model.classifier = nn.Identity()
        num_ftrs = 1536
    elif network == 'bagnet33':
        print("BagNet not available, using resnet50")
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Identity()
        num_ftrs = 2048
    else:
        raise ValueError(f'Unknown network: {network}')
    
    if cfg.conv_cls:
        model = FullyConvModel(cfg, model, num_classes)
    else:
        classifier = nn.Linear(num_ftrs, num_classes)
        model = nn.Sequential(model, classifier)
    
    return model


# =============================================================================
# LOSS, OPTIMIZER, SCHEDULER
# =============================================================================
def get_loss_fn(cfg):
    if cfg.loss == 'ce':
        if cfg.label_smoothing > 0:
            return nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
        return nn.CrossEntropyLoss()
    elif cfg.loss == 'focal':
        alpha = torch.ones(cfg.num_classes).to(cfg.device)
        return FocalLoss(alpha=alpha, gamma=2.0, reduction='mean')
    elif cfg.loss == 'kappa':
        return KappaLoss(num_classes=cfg.num_classes)
    else:
        raise ValueError(f'Unknown loss: {cfg.loss}')


def get_optimizer(cfg, model):
    if cfg.optimizer == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f'Unknown optimizer: {cfg.optimizer}')


def get_scheduler(cfg, optimizer, num_epochs):
    if cfg.scheduler == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=cfg.lr * 0.01)
    elif cfg.scheduler == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs//3, gamma=0.1)
    elif cfg.scheduler == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    return None


def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


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
def train_epoch(cfg, model, train_loader, optimizer, criterion, epoch):
    model.train()
    epoch_loss = 0.0
    all_preds = []
    all_targets = []
    
    for batch_idx, batch in enumerate(train_loader):
        images = batch['image'].to(cfg.device)
        targets = batch['target'].to(cfg.device)
        
        # MixUp
        if cfg.mixup > 0 and np.random.rand() < 0.5:
            images, targets_a, targets_b, lam = mixup_data(images, targets, cfg.mixup)
            optimizer.zero_grad()
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            optimizer.zero_grad()
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, targets)
        
        loss.backward()
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        
        epoch_loss += loss.item()
        all_preds.append(outputs.detach())
        all_targets.append(targets.detach())
        
        if batch_idx % cfg.log_interval == 0:
            print(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}')
    
    train_metrics = compute_all_metrics(torch.cat(all_preds), torch.cat(all_targets), cfg.num_classes)
    train_metrics['loss'] = epoch_loss / len(train_loader)
    return train_metrics


def evaluate(cfg, model, loader, criterion):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(cfg.device)
            targets = batch['target'].to(cfg.device)
            
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            all_preds.append(outputs)
            all_targets.append(targets)
    
    metrics = compute_all_metrics(torch.cat(all_preds), torch.cat(all_targets), cfg.num_classes)
    metrics['loss'] = total_loss / len(loader)
    return metrics


# =============================================================================
# HEATMAP GENERATION (GradCAM-style)
# =============================================================================
def generate_heatmaps(cfg, model, save_path, num_samples=20):
    if not cfg.conv_cls:
        print("Heatmaps only available with conv_cls=True")
        return
    
    # Get underlying model if DataParallel
    model_to_use = model.module if isinstance(model, nn.DataParallel) else model
    model_to_use.eval()
    
    val_csv_path = os.path.join(cfg.data_dir, cfg.train_csv)
    val_img_dir = os.path.join(cfg.data_dir, cfg.train_img_dir)
    val_df = pd.read_csv(val_csv_path).sample(n=min(num_samples, 100), random_state=42)
    
    transform = get_transforms(cfg.img_size, 'val')
    
    heatmap_dir = os.path.join(save_path, 'heatmaps')
    os.makedirs(heatmap_dir, exist_ok=True)
    
    import cv2
    
    for _, row in val_df.iterrows():
        img_id = row['id_code']
        label = row['diagnosis']
        
        img_path = os.path.join(val_img_dir, f"{img_id}.png")
        image = Image.open(img_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(cfg.device)
        
        with torch.no_grad():
            outputs, activations = model_to_use(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1).item()
        
        act_map = activations[0, pred].cpu().numpy()
        act_map = cv2.resize(act_map, (image.width, image.height))
        act_map = (act_map - act_map.min()) / (act_map.max() - act_map.min() + 1e-8)
        
        heatmap = cv2.applyColorMap(np.uint8(255 * act_map), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        orig = np.array(image.resize((512, 512)))
        heatmap_resized = cv2.resize(heatmap, (512, 512))
        overlay = cv2.addWeighted(orig, 0.6, heatmap_resized, 0.4, 0)
        
        cv2.imwrite(os.path.join(heatmap_dir, f"{img_id}_gt{label}_pred{pred}.png"), 
                   cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    print(f"Heatmaps saved to {heatmap_dir}")


# =============================================================================
# SUBMISSION GENERATION
# =============================================================================
def generate_submission(cfg, model, save_path):
    # Get underlying model if DataParallel
    model_to_use = model.module if isinstance(model, nn.DataParallel) else model
    model_to_use.eval()
    
    test_csv_path = os.path.join(cfg.data_dir, cfg.test_csv)
    test_img_dir = os.path.join(cfg.data_dir, cfg.test_img_dir)
    test_df = pd.read_csv(test_csv_path)
    
    transform = get_transforms(cfg.img_size, 'val')
    test_dataset = APTOSDataset(test_csv_path, test_img_dir, transform=transform, mode='val')
    test_loader = DataLoader(test_dataset, batch_size=cfg.val_batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)
    
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(cfg.device)
            outputs = model_to_use(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            probs = torch.softmax(outputs, dim=1)
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
        
        # Training - scale batch size by num GPUs
        self.batch_size = args.batch_size * max(1, self.num_gpus)
        self.val_batch_size = args.val_batch_size * max(1, self.num_gpus)
        self.num_workers = args.workers
        self.persistent_workers = True if args.workers > 0 else False
        self.num_classes = args.num_classes
        
        # Model
        self.network = args.network
        self.pretrained = args.pretrained
        self.conv_cls = args.conv_cls
        self.loss = args.loss
        self.epochs = args.epochs
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.optimizer = args.optimizer
        self.scheduler = args.scheduler
        self.mixup = args.mixup
        self.cutmix = args.cutmix
        self.label_smoothing = args.label_smoothing
        self.grad_clip = args.grad_clip
        self.log_interval = args.log_interval
        
        # Output
        self.save_path = args.output
        self.generate_heatmaps = args.generate_heatmaps
        
        # ============================================================
        # COMPATIBILITY LAYER for original SoftCAM modules
        # They expect: cfg.train.network, cfg.data.num_classes, etc.
        # ============================================================
        class TrainConfig:
            def __init__(self, cfg):
                self.network = cfg.network
                self.pretrained = cfg.pretrained
                self.conv_cls = cfg.conv_cls
                self.loss = cfg.loss
        
        class DataConfig:
            def __init__(self, cfg):
                self.num_classes = cfg.num_classes
                self.img_size = cfg.img_size
                self.batch_size = cfg.batch_size
        
        self.train = TrainConfig(self)
        self.data = DataConfig(self)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_save_paths(cfg):
    save_path = os.path.join(cfg.save_path, cfg.network, datetime.now().strftime("%Y%m%d-%H%M%S"))
    return save_path


# =============================================================================
# MAIN
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='SoftCAM APTOS 2019 Training')
    
    # Data paths (Kaggle default)
    parser.add_argument('--data-dir', default='/kaggle/input/competitions/aptos2019-blindness-detection', type=str)
    parser.add_argument('--train-csv', default='train.csv', type=str)
    parser.add_argument('--test-csv', default='test.csv', type=str)
    parser.add_argument('--train-img-dir', default='train_images', type=str)
    parser.add_argument('--test-img-dir', default='test_images', type=str)
    
    # Model
    parser.add_argument('--network', default='resnet50', type=str, 
                        choices=['resnet50', 'resnet101', 'resnet152', 'densenet121', 'densenet169', 'efficientnet_b0', 'efficientnet_b3'])
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false')
    parser.add_argument('--conv-cls', action='store_true', default=True, help='Use conv classifier for heatmaps')
    parser.add_argument('--no-conv-cls', dest='conv_cls', action='store_false')
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
    parser.add_argument('--loss', default='ce', type=str, choices=['ce', 'focal', 'kappa'])
    parser.add_argument('--label-smoothing', default=0.1, type=float)
    
    # Regularization
    parser.add_argument('--mixup', default=0.5, type=float)
    parser.add_argument('--cutmix', default=1.0, type=float)
    parser.add_argument('--grad-clip', default=1.0, type=float)
    
    # System
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--output', default='/kaggle/working/softcam_aptos', type=str)
    parser.add_argument('--log-interval', default=20, type=int)
    parser.add_argument('--generate-heatmaps', action='store_true', default=True)
    
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config(args)
    
    # Print config - NO driver detection, just is_available
    print("="*60)
    print("SOFTCAM APTOS TRAINING CONFIG")
    print("="*60)
    print(f"Device: {cfg.device}")
    print(f"GPUs available: {cfg.num_gpus}")
    if cfg.num_gpus > 0:
        print(f"  Using {cfg.num_gpus} GPU(s)")
    print(f"Network: {cfg.network}")
    print(f"Conv Classifier (heatmaps): {cfg.conv_cls}")
    print(f"Epochs: {cfg.epochs}")
    print(f"Batch Size: {cfg.batch_size} (per GPU: {args.batch_size})")
    print(f"Workers: {cfg.num_workers} (persistent: {cfg.persistent_workers})")
    print(f"Loss: {cfg.loss}")
    print(f"MixUp: {cfg.mixup}, CutMix: {cfg.cutmix}")
    print(f"Output: {cfg.save_path}")
    print("="*60)
    
    # Set seed
    set_random_seed(args.seed)
    
    # Save path
    save_path = load_save_paths(cfg)
    os.makedirs(save_path, exist_ok=True)
    print(f'Save path: {save_path}')
    
    # TensorBoard
    logger = SummaryWriter(save_path)
    
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
    
    # Model
    model = generate_model(cfg)
    model.to(cfg.device)
    
    # Multi-GPU: wrap with DataParallel
    if cfg.num_gpus > 1:
        print(f"Using DataParallel on {cfg.num_gpus} GPUs")
        model = nn.DataParallel(model)
    
    print(f'Model: {cfg.network}, Parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Loss, Optimizer, Scheduler
    criterion = get_loss_fn(cfg)
    optimizer = get_optimizer(cfg, model)
    scheduler = get_scheduler(cfg, optimizer, cfg.epochs)
    
    # Training
    print('\nStarting training...')
    best_metric = 0.0
    best_epoch = 0
    
    for epoch in range(cfg.epochs):
        epoch_start = time.time()
        
        train_metrics = train_epoch(cfg, model, train_loader, optimizer, criterion, epoch)
        val_metrics = evaluate(cfg, model, val_loader, criterion)
        
        # Logging
        logger.add_scalar('Loss/train', train_metrics['loss'], epoch)
        logger.add_scalar('Loss/val', val_metrics['loss'], epoch)
        logger.add_scalar('AUC/train', train_metrics['auc_macro'], epoch)
        logger.add_scalar('AUC/val', val_metrics['auc_macro'], epoch)
        logger.add_scalar('Acc/train', train_metrics['accuracy'], epoch)
        logger.add_scalar('Acc/val', val_metrics['accuracy'], epoch)
        logger.add_scalar('QWK/val', val_metrics['qwk'], epoch)
        logger.add_scalar('Kappa/val', val_metrics['kappa'], epoch)
        
        # Print metrics
        print(f'\nEpoch {epoch} ({time.time() - epoch_start:.1f}s):')
        print(f'  Train: Loss={train_metrics["loss"]:.4f}, QWK={train_metrics["qwk"]:.4f}, Acc={train_metrics["accuracy"]:.4f}, AUC={train_metrics["auc_macro"]:.4f}')
        print(f'  Valid: Loss={val_metrics["loss"]:.4f}, QWK={val_metrics["qwk"]:.4f}, Acc={val_metrics["accuracy"]:.4f}, AUC={val_metrics["auc_macro"]:.4f}, Kappa={val_metrics["kappa"]:.4f}')
        print(f'  AUC per class: {val_metrics["auc_per_class"]}')
        print(f'  Acc per class: {val_metrics["accuracy_per_class"]}')
        print(f'  Confusion Matrix:')
        print(val_metrics['confusion_matrix'])
        
        # Save best model (by QWK - official APTOS metric)
        metric = val_metrics['qwk']
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pt'))
            print(f'  -> Best model saved (QWK: {best_metric:.4f})')
        
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(metric)
            else:
                scheduler.step()
    
    print(f'\n{"="*60}')
    print(f'Best validation QWK: {best_metric:.4f} at epoch {best_epoch}')
    print(f'{"="*60}')
    
    # Load best model for final evaluation
    best_model_path = os.path.join(save_path, 'best_model.pt')
    state_dict = torch.load(best_model_path, map_location=cfg.device)
    
    # Handle DataParallel state dict
    if cfg.num_gpus > 1 and not isinstance(model, nn.DataParallel):
        # Model wasn't wrapped but state dict has module. prefix
        model = nn.DataParallel(model)
        model.load_state_dict(state_dict)
    elif cfg.num_gpus == 1 and isinstance(model, nn.DataParallel):
        # Model was wrapped but now single GPU
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    
    # Final validation with best model
    print('\n=== Final Evaluation (Best Model) ===')
    final_metrics = evaluate(cfg, model, val_loader, criterion)
    print(f"FINAL: QWK={final_metrics['qwk']:.4f}, Acc={final_metrics['accuracy']:.4f}, AUC={final_metrics['auc_macro']:.4f}")
    print(f"Confusion Matrix:\n{final_metrics['confusion_matrix']}")
    
    # Generate heatmaps with best model
    if args.generate_heatmaps:
        print('\nGenerating heatmaps...')
        generate_heatmaps(cfg, model, save_path, num_samples=20)
    
    # Generate submission
    print('\nGenerating submission...')
    generate_submission(cfg, model, save_path)
    
    logger.close()
    print('\nDone!')
    return model, save_path


if __name__ == '__main__':
    main()