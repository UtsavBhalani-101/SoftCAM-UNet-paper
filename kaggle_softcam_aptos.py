#!/usr/bin/env python3
"""
Kaggle Notebook: SoftCAM for APTOS 2019 Blindness Detection
5-class Diabetic Retinopathy Classification
Repository: SoftCAM (weakly-supervised localization framework)
"""

# =============================================================================
# INSTALL DEPENDENCIES (run once in Kaggle)
# =============================================================================
# !pip install -q timm albumentations torchmetrics pyyaml tensorboard

# =============================================================================
# IMPORTS
# =============================================================================
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import torchvision.transforms as T
from torchvision import models
from PIL import Image

from torchmetrics.functional.classification import multiclass_auroc, multiclass_accuracy, multiclass_cohen_kappa

# Add SoftCAM path
sys.path.append('/kaggle/working/SoftCAM')
from modules.models import FullyConvModel, ConvClassifier
from modules.loss import FocalLoss, KappaLoss
from utils.func import one_hot
from utils.metrics import Estimator
from data.builder import generate_dataset  # We'll create our own dataset instead

# =============================================================================
# APTOS DATASET FOR SOFTCAM
# =============================================================================
class APTOSDatasetSoftCAM(Dataset):
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
            
        # SoftCAM expects dict with 'image' and 'target'
        return {'image': image, 'target': torch.tensor(label, dtype=torch.long)}


def get_aptos_transforms_softcam(img_size=512, mode='train'):
    """SoftCAM uses different transform pipeline"""
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
# CONFIGURATION CLASS (replaces YAML config)
# =============================================================================
class Config:
    def __init__(self):
        # Base
        self.base = type('', (), {})()
        self.base.dataset = 'APTOS'
        self.base.random_seed = 42
        self.base.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base.num_gpus = 1
        
        # Data
        self.data = type('', (), {})()
        self.data.data_dir = '/kaggle/input/competitions/aptos2019-blindness-detection'
        self.data.train_csv = 'train.csv'
        self.data.test_csv = 'test.csv'
        self.data.train_img_dir = 'train_images'
        self.data.test_img_dir = 'test_images'
        self.data.img_size = 512
        self.data.batch_size = 16
        self.data.val_batch_size = 32
        self.data.num_workers = 4
        self.data.num_classes = 5
        self.data.binary = False  # Multi-class
        self.data.target = 'diagnosis'
        
        # Train
        self.train = type('', (), {})()
        self.train.network = 'resnet50'  # Options: resnet50, resnet101, densenet121, efficientnet_b0, bagnet33
        self.train.pretrained = True
        self.train.conv_cls = True  # Use conv classifier for heatmaps
        self.train.lambda_l1 = 0.0
        self.train.lambda_l2 = 0.0
        self.train.loss = 'ce'  # 'ce', 'focal', 'kappa'
        self.train.epochs = 50
        self.train.lr = 1e-4
        self.train.weight_decay = 1e-4
        self.train.optimizer = 'adamw'
        self.train.scheduler = 'cosine'
        self.train.warmup_epochs = 5
        self.train.mixup = 0.5
        self.train.cutmix = 1.0
        self.train.label_smoothing = 0.1
        self.train.grad_clip = 1.0
        
        # Output
        self.dset = type('', (), {})()
        self.dset.save_path = '/kaggle/working/softcam_output'
        self.dset.root = self.data.data_dir


cfg = Config()

# =============================================================================
# UTILS (from SoftCAM utils/func.py)
# =============================================================================
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_save_paths(cfg):
    save_path = os.path.join(cfg.dset.save_path, cfg.train.network, 
                            datetime.now().strftime("%Y%m%d-%H%M%S"))
    return save_path

def copy_file(folders, save_path):
    # Skip in Kaggle
    pass

def print_config(config_dict):
    for k, v in config_dict.items():
        print(f'{k}: {v}')

def print_msg(msg):
    print(f'[INFO] {msg}')

# =============================================================================
# MODEL GENERATION (from SoftCAM modules/builder.py)
# =============================================================================
def generate_model(cfg):
    network = cfg.train.network
    num_classes = cfg.data.num_classes
    pretrained = cfg.train.pretrained
    
    # Load backbone
    if network == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Identity()  # Remove classifier
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
        # BagNet not in torchvision, use resnet50 as fallback
        print("BagNet not available, using resnet50")
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Identity()
        num_ftrs = 2048
    else:
        raise ValueError(f'Unknown network: {network}')
    
    if cfg.train.conv_cls:
        # Fully convolutional model for heatmap generation
        model = FullyConvModel(cfg, model, num_classes)
    else:
        # Standard classifier
        classifier = nn.Linear(num_ftrs, num_classes)
        model = nn.Sequential(model, classifier)
    
    return model


# =============================================================================
# LOSS FUNCTION
# =============================================================================
def get_loss_fn(cfg):
    if cfg.train.loss == 'ce':
        if cfg.train.label_smoothing > 0:
            return nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)
        return nn.CrossEntropyLoss()
    elif cfg.train.loss == 'focal':
        alpha = torch.ones(cfg.data.num_classes).to(cfg.base.device)
        return FocalLoss(alpha=alpha, gamma=2.0, reduction='mean')
    elif cfg.train.loss == 'kappa':
        return KappaLoss(num_classes=cfg.data.num_classes)
    else:
        raise ValueError(f'Unknown loss: {cfg.train.loss}')


# =============================================================================
# OPTIMIZER & SCHEDULER
# =============================================================================
def get_optimizer(cfg, model):
    if cfg.train.optimizer == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, 
                                 weight_decay=cfg.train.weight_decay)
    elif cfg.train.optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=cfg.train.lr, 
                               momentum=0.9, weight_decay=cfg.train.weight_decay)
    else:
        raise ValueError(f'Unknown optimizer: {cfg.train.optimizer}')

def get_scheduler(cfg, optimizer, num_epochs):
    if cfg.train.scheduler == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, 
                                                          eta_min=cfg.train.lr * 0.01)
    elif cfg.train.scheduler == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs//3, gamma=0.1)
    elif cfg.train.scheduler == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                          patience=5, factor=0.5)
    return None


# =============================================================================
# MIXUP / CUTMIX
# =============================================================================
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
# TRAIN & EVALUATE
# =============================================================================
def train(cfg, model, train_loader, val_loader, optimizer, scheduler, criterion, 
          estimator, logger, save_path):
    
    model.to(cfg.base.device)
    best_metric = 0.0
    best_epoch = 0
    
    for epoch in range(cfg.train.epochs):
        model.train()
        epoch_loss = 0.0
        all_preds = []
        all_targets = []
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(cfg.base.device)
            targets = batch['target'].to(cfg.base.device)
            
            # Mixup
            if cfg.train.mixup > 0 and np.random.rand() < 0.5:
                images, targets_a, targets_b, lam = mixup_data(images, targets, cfg.train.mixup)
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
            if cfg.train.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            
            epoch_loss += loss.item()
            all_preds.append(outputs.detach())
            all_targets.append(targets.detach())
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}')
        
        # Validation
        val_metrics = evaluate(cfg, model, val_loader, criterion, estimator, 'validation')
        train_metrics = compute_metrics(torch.cat(all_preds), torch.cat(all_targets), cfg.data.num_classes)
        
        # Logging
        logger.add_scalar('Loss/train', epoch_loss / len(train_loader), epoch)
        logger.add_scalar('Loss/val', val_metrics['loss'], epoch)
        logger.add_scalar('AUC/train', train_metrics['auc_mean'], epoch)
        logger.add_scalar('AUC/val', val_metrics['auc_mean'], epoch)
        logger.add_scalar('Acc/train', train_metrics['acc'], epoch)
        logger.add_scalar('Acc/val', val_metrics['acc'], epoch)
        logger.add_scalar('Kappa/val', val_metrics.get('kappa', 0), epoch)
        
        print(f'Epoch {epoch}: Train Loss={epoch_loss/len(train_loader):.4f}, '
              f'Val Loss={val_metrics["loss"]:.4f}, Val AUC={val_metrics["auc_mean"]:.4f}, '
              f'Val Acc={val_metrics["acc"]:.4f}, Val Kappa={val_metrics.get("kappa", 0):.4f}')
        
        # Save best model
        metric = val_metrics['auc_mean']
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pt'))
            print(f'  -> Best model saved (AUC: {best_metric:.4f})')
        
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(metric)
            else:
                scheduler.step()
    
    print(f'Best validation AUC: {best_metric:.4f} at epoch {best_epoch}')
    return best_metric


def evaluate(cfg, model, loader, criterion, estimator, type_ds='validation'):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(cfg.base.device)
            targets = batch['target'].to(cfg.base.device)
            
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            all_preds.append(outputs)
            all_targets.append(targets)
    
    metrics = compute_metrics(torch.cat(all_preds), torch.cat(all_targets), cfg.data.num_classes)
    metrics['loss'] = total_loss / len(loader)
    return metrics


def compute_metrics(preds, targets, num_classes):
    """Compute AUC, Accuracy, Kappa"""
    preds = preds.cpu()
    targets = targets.cpu()
    
    auc = multiclass_auroc(preds, targets, num_classes=num_classes, average=None)
    auc_mean = multiclass_auroc(preds, targets, num_classes=num_classes, average='macro')
    acc = multiclass_accuracy(preds, targets, num_classes=num_classes, average='macro')
    kappa = multiclass_cohen_kappa(preds, targets, num_classes=num_classes)
    
    return {
        'auc': auc.numpy(),
        'auc_mean': auc_mean.item(),
        'acc': acc.item(),
        'kappa': kappa.item()
    }


# =============================================================================
# SUBMISSION GENERATION
# =============================================================================
def generate_submission(cfg, model, save_path):
    model.eval()
    
    test_csv_path = os.path.join(cfg.data.data_dir, cfg.data.test_csv)
    test_img_dir = os.path.join(cfg.data.data_dir, cfg.data.test_img_dir)
    test_df = pd.read_csv(test_csv_path)
    
    transform = get_aptos_transforms_softcam(cfg.data.img_size, 'val')
    test_dataset = APTOSDatasetSoftCAM(test_csv_path, test_img_dir, transform=transform, mode='val')
    test_loader = DataLoader(test_dataset, batch_size=cfg.data.val_batch_size, 
                             shuffle=False, num_workers=cfg.data.num_workers, pin_memory=True)
    
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(cfg.base.device)
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            predictions.extend(preds.cpu().numpy())
    
    submission = pd.DataFrame({'id_code': test_df['id_code'], 'diagnosis': predictions})
    submission_path = os.path.join(save_path, 'submission.csv')
    submission.to_csv(submission_path, index=False)
    print(f'Submission saved to {submission_path}')
    print(submission['diagnosis'].value_counts().sort_index())


# =============================================================================
# MAIN
# =============================================================================
def main():
    # Setup
    set_random_seed(cfg.base.random_seed)
    save_path = load_save_paths(cfg)
    os.makedirs(save_path, exist_ok=True)
    print(f'Save path: {save_path}')
    
    logger = SummaryWriter(save_path)
    
    # Data
    train_transform = get_aptos_transforms_softcam(cfg.data.img_size, 'train')
    val_transform = get_aptos_transforms_softcam(cfg.data.img_size, 'val')
    
    train_csv_path = os.path.join(cfg.data.data_dir, cfg.data.train_csv)
    train_img_dir = os.path.join(cfg.data.data_dir, cfg.data.train_img_dir)
    
    full_dataset = APTOSDatasetSoftCAM(train_csv_path, train_img_dir, transform=train_transform, mode='train')
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transform
    val_dataset.dataset.mode = 'val'
    
    print(f'Train: {len(train_dataset)}, Val: {len(val_dataset)}')
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True,
                              num_workers=cfg.data.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.data.val_batch_size, shuffle=False,
                            num_workers=cfg.data.num_workers, pin_memory=True)
    
    # Model
    model = generate_model(cfg)
    print(f'Model: {cfg.train.network}, Parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Loss, Optimizer, Scheduler
    criterion = get_loss_fn(cfg)
    optimizer = get_optimizer(cfg, model)
    scheduler = get_scheduler(cfg, optimizer, cfg.train.epochs)
    
    # Estimator (for metrics)
    estimator = Estimator(cfg) if hasattr(cfg, 'Estimator') else None
    
    # Train
    print('Starting training...')
    train(cfg, model, train_loader, val_loader, optimizer, scheduler, criterion, 
          estimator, logger, save_path)
    
    # Generate submission with best model
    print('\nLoading best model for submission...')
    model.load_state_dict(torch.load(os.path.join(save_path, 'best_model.pt')))
    generate_submission(cfg, model, save_path)
    
    logger.close()
    print('Done!')


if __name__ == '__main__':
    main()