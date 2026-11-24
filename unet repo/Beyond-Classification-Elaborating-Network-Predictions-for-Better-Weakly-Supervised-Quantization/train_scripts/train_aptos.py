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
from sklearn.exceptions import UndefinedMetricWarning


import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from torch.utils.data import Dataset, DataLoader, random_split, Subset, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import warnings

import torchvision.transforms as T
from PIL import Image

from torchmetrics.functional.classification import multiclass_auroc, multiclass_accuracy, multiclass_cohen_kappa

# =============================================================================
# IMPORT EXISTING CHEN'S UNET MODEL CODE
# =============================================================================
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from models.model_segrdnet import SegRDNet

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


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
    ce_losses_m = 0.0
    aux_losses_m = 0.0
    reg_losses_m = 0.0
    num_batches = 0

    model.train()
    optimizer.zero_grad()
    
    grad_accum_steps = getattr(cfg, 'grad_accum_steps', 1)
    
    # Check if the underlying model (DDP/DataParallel-unwrapped) has SoftCAM
    # regularization; nn.Module attribute lookups do not forward methods through
    # the DDP wrapper, so hasattr on the wrapper would silently return False
    base_model = _unwrap_model(model)
    has_softcam = hasattr(base_model, 'get_softcam_regularization')
    
    for batch_idx, (input, target) in enumerate(loader):
        input = input.to(device)
        target = target.to(device)
        
        # Mixed precision forward with separately tracked loss terms
        with amp_autocast():
            if cfg.use_softcam_decoder0:
                output, _, scale_logits = model(input, eval_mode=False)
                # SegRDNet returns scale logits as (decoder3, decoder2,
                # decoder1, decoder0). Supervise the new decoder0 SoftCAM head
                # so it learns class evidence, not just small maps.
                ce_loss = loss_fn(output, target)
                softcam_aux_loss = loss_fn(scale_logits[-1], target)
                loss = ce_loss + cfg.softcam_aux_weight * softcam_aux_loss
            else:
                output = model(input)
                ce_loss = loss_fn(output, target)
                softcam_aux_loss = None
                loss = ce_loss
            
            # Add SoftCAM regularization loss (L1/L2 on evidence maps)
            softcam_reg = None
            if has_softcam:
                softcam_reg = base_model.get_softcam_regularization()
                if softcam_reg != 0.0:
                    loss = loss + softcam_reg
        
        ce_losses_m += ce_loss.item()
        if softcam_aux_loss is not None:
            aux_losses_m += cfg.softcam_aux_weight * softcam_aux_loss.item()
        if softcam_reg is not None:
            reg_losses_m += float(softcam_reg)
        
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
        
        if batch_idx % cfg.log_interval == 0 and getattr(cfg, 'is_main', True):
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
    
    metrics = global_metrics(p_test, y_test, cfg)
    if dist.is_available() and dist.is_initialized():
        # Average per-rank epoch losses into one global number on every rank
        loss_t = torch.tensor([avg_loss], device=p_test.device)
        dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)
        avg_loss = loss_t.item()
    metrics['loss'] = avg_loss
    
    if getattr(cfg, 'is_main', True):
        nb = max(1, num_batches)
        terms = [f'ce={ce_losses_m / nb:.4f}']
        if cfg.use_softcam_decoder0:
            terms.append(f'softcam-aux={aux_losses_m / nb:.4f}')
        if reg_losses_m:
            terms.append(f'softcam-reg={reg_losses_m / nb:.4f}')
        print(f"  Train: Loss={metrics['loss']:.4f} ({', '.join(terms)}), "
              f"QWK={metrics['qwk']:.4f}, Acc={metrics['accuracy']:.4f}, AUC={metrics['auc_macro']:.4f}")
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
    
    metrics = global_metrics(p_test, y_test, cfg)
    if dist.is_available() and dist.is_initialized():
        loss_t = torch.tensor([avg_loss], device=p_test.device)
        dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)
        avg_loss = loss_t.item()
    metrics['loss'] = avg_loss
    
    if getattr(cfg, 'is_main', True):
        print(f"  Valid: Loss={metrics['loss']:.4f}, QWK={metrics['qwk']:.4f}, Acc={metrics['accuracy']:.4f}, AUC={metrics['auc_macro']:.4f}, Kappa={metrics['kappa']:.4f}")
        print(f"  AUC per class: {metrics['auc_per_class']}")
        print(f"  Acc per class: {metrics['accuracy_per_class']}")
        pred_counts = np.bincount(metrics['pred_labels'], minlength=cfg.num_classes)
        print(f"  Val images predicted per class: {pred_counts.tolist()}")
        for grade in range(cfg.num_classes):
            if pred_counts[grade] == 0:
                print(f'  WARNING: grade {grade} never predicted')
        print(f"  Confusion Matrix:\n{metrics['confusion_matrix']}")
    
    return metrics


# =============================================================================
# DEBUG MODES & WEIGHT STATUS
# =============================================================================
def pretrained_weights_status(base_model, num_classes):
    """Verify the timm rdnet_tiny backbone actually carries ImageNet weights.

    The SegRDNet constructor hardcodes create_model('rdnet_tiny', pretrained=True);
    this compares one backbone tensor against a from-scratch init to confirm
    pretrained weights really differ from random init.
    """
    status = f'Pretrained backbone: timm rdnet_tiny (pretrained=True)'
    try:
        from models.rdnet import rdnet_tiny
        scratch = rdnet_tiny(pretrained=False, num_classes=num_classes)
        w_scratch = next(iter(scratch.state_dict().values())).detach().cpu()
        w_model = next(iter(base_model.model.state_dict().values())).detach().cpu()
        status += ': ImageNet weights loaded' if not torch.equal(w_scratch, w_model) \
            else ': weights match a from-scratch init (pretrained weights may NOT have loaded)'
    except Exception as exc:
        status += f': could not verify ({exc})'
    return status


def tiny_overfit(cfg, model, optimizer, loss_fn, amp_autocast=suppress, loss_scaler=None):
    """Overfit 64 val-transform images for 50 optimizer steps; report train accuracy.

    Deliberately uses val transforms so accuracy is meaningful and overfitting
    is achievable; if the loss does not collapse, something is broken.
    """
    csv_path = os.path.join(cfg.data_dir, cfg.train_csv)
    img_dir = os.path.join(cfg.data_dir, cfg.train_img_dir)
    dataset = APTOSDataset(csv_path, img_dir, transform=get_transforms(cfg.img_size, 'val'), mode='val')
    n = min(64, len(dataset))
    loader = DataLoader(
        Subset(dataset, list(range(n))),
        batch_size=min(cfg.batch_size, n),
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # The 64 images are the first rows of train.csv, which is not class-balanced;
    # if these are mostly grade 0, a collapsed (majority-class) model already
    # scores that share as accuracy. Print the composition to interpret the cap.
    subset_labels = np.bincount([int(dataset[i][1]) for i in range(n)], minlength=cfg.num_classes)
    print(f'Tiny-overfit: {n} images (val transforms), 50 optimizer steps')
    print(f'  class distribution of those {n} images: {subset_labels.tolist()}')

    # Diagnostic plumbing test: the training lr (1e-4) is far too small to
    # collapse the loss in only 50 steps on this fresh 4-decoder stack. Raise
    # it so the mode can actually prove gradients flow; the run exits right
    # after, so nothing downstream is affected.
    overfit_lr = max(cfg.lr, 1e-3)
    for group in optimizer.param_groups:
        group['lr'] = overfit_lr
    print(f'  tiny-overfit LR override: {cfg.lr} -> {overfit_lr}')
    model.train()
    steps, correct, seen, running_loss = 0, 0, 0, 0.0
    while steps < 50:
        for input, target in loader:
            if steps >= 50:
                break
            input = input.to(cfg.device)
            target = target.to(cfg.device)
            optimizer.zero_grad()
            with amp_autocast():
                if cfg.use_softcam_decoder0:
                    output, _, scale_logits = model(input, eval_mode=False)
                    loss = loss_fn(output, target) + cfg.softcam_aux_weight * loss_fn(scale_logits[-1], target)
                else:
                    output = model(input)
                    loss = loss_fn(output, target)
            if loss_scaler is not None:
                loss_scaler.scale(loss).backward()
                if cfg.grad_clip:
                    loss_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                loss_scaler.step(optimizer)
                loss_scaler.update()
            else:
                loss.backward()
                if cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
            preds = output.argmax(dim=1)
            correct += (preds == target).sum().item()
            seen += target.size(0)
            running_loss += loss.item()
            steps += 1
            if steps % 10 == 0:
                print(f'  step {steps:02d}/50 | running loss {running_loss / steps:.3f} '
                      f'| running train acc {correct / max(1, seen):.3f}')

    print(f'tiny-overfit training accuracy over all 50 steps: '
          f'{correct / max(1, seen):.4f} (mean loss {running_loss / steps:.4f})')


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
        # One process per GPU (DDP via torchrun); plain launch = single process
        self.rank, self.world_size, self.device = init_distributed()
        self.is_main = self.rank == 0
        self.num_gpus = self.world_size if self.device.type == 'cuda' else 0
        
        # Data paths
        self.data_dir = args.data_dir
        self.train_csv = args.train_csv
        self.test_csv = args.test_csv
        self.train_img_dir = args.train_img_dir
        self.test_img_dir = args.test_img_dir
        self.img_size = args.img_size
        
        # Per-GPU batch size: SegRDNet hooks prevent DataParallel, so with DDP
        # each process runs its own replica; cap per-step batch at 8 for 512x512
        # on T4 16GB and compensate with gradient accumulation across the fleet.
        self.batch_size = min(args.batch_size, 8)
        self.val_batch_size = min(args.val_batch_size, 16)
        
        # Gradient accumulation to maintain the target effective batch size
        self.grad_accum_steps = max(1, round(args.batch_size / (self.batch_size * self.world_size)))
        self.num_workers = args.workers
        self.persistent_workers = True if args.workers > 0 else False
        self.num_classes = args.num_classes
        self.pooling = getattr(args, 'pooling', 'avg')
        
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
        self.softcam_decoder0_mode = args.softcam_decoder0_mode
        self.softcam_lambda_l1 = args.softcam_lambda_l1
        self.softcam_lambda_l2 = args.softcam_lambda_l2
        self.softcam_aux_weight = args.softcam_aux_weight
        
        # AMP
        self.amp = args.amp
        self.amp_dtype = args.amp_dtype
        
        # Debug / sanity modes
        self.smoke = getattr(args, 'smoke', False)
        self.tiny_overfit = getattr(args, 'tiny_overfit', False)
        
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
# WANDB & CHECKPOINT HELPERS
# =============================================================================
def per_class_recall(conf_mat):
    """Recall per class from a confusion matrix (0 when a class has no GT samples)."""
    support = conf_mat.sum(axis=1)
    return np.divide(np.diag(conf_mat), support, out=np.zeros_like(np.diag(conf_mat), dtype=float), where=support > 0)


def _unwrap_model(model):
    """Return the underlying module for DDP/DataParallel wrappers."""
    if isinstance(model, (NativeDDP, nn.DataParallel)):
        return model.module
    return model


def init_distributed():
    """Initialize one process per GPU (launched via torchrun / torch.distributed.run).

    torchrun sets RANK/WORLD_SIZE/LOCAL_RANK; with a plain `python` launch this
    falls back to single-process execution on cuda:0.
    """
    if dist.is_available() and not dist.is_initialized() and 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend='nccl')
        else:
            dist.init_process_group(backend='gloo')
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0))) if torch.cuda.is_available() \
            else torch.device('cpu')
        name = torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'
        # Every rank prints, so the log itself proves how many processes/GPUs ran
        print(f"[rank {rank}/{world_size}] using {device} ({name}); "
              f"visible cuda devices: {torch.cuda.device_count()}", flush=True)
    else:
        rank, world_size = 0, 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return rank, world_size, device


def gather_tensor(tensor):
    """All-gather variable-sized tensors (padded) and concatenate; identical result on every rank."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor
    local_n = torch.tensor([tensor.shape[0]], device=tensor.device)
    sizes = [torch.zeros_like(local_n) for _ in range(world_size)]
    dist.all_gather(sizes, local_n)
    max_n = int(max(s.item() for s in sizes))
    if max_n > tensor.shape[0]:
        fill = -1 if tensor.dtype == torch.long else 0.0
        pad = tensor.new_full((max_n - tensor.shape[0], *tensor.shape[1:]), fill)
        tensor = torch.cat([tensor, pad], dim=0)
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor.contiguous())
    return torch.cat(gathered, dim=0)


def global_metrics(p_test, y_test, cfg):
    """Concatenate local predictions/targets across ranks and compute dataset-wide metrics."""
    y_test = gather_tensor(y_test)
    p_test = gather_tensor(p_test)
    valid = y_test >= 0  # padded slots only exist in the target tensor
    y_test = y_test[valid]
    p_test = p_test[valid[:p_test.shape[0]]]
    return compute_all_metrics(p_test, y_test, cfg.num_classes)


def init_wandb(args, resume_state=None):
    """Start a wandb run when --wandb is set; on resume, continue the same run id."""
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit('--wandb requires the wandb package: pip install wandb') from exc
    run_id = resume_state.get('wandb_run_id') if resume_state else None
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config=vars(args),
        id=run_id,
        resume='must' if run_id else None,
    )
    return run


def wandb_log_epoch(run, epoch, lr, secs, train_loss, val_loss, train_metrics, val_metrics, num_classes):
    if run is None:
        return
    import wandb
    cm = val_metrics['confusion_matrix']
    recall = per_class_recall(cm)
    pred_counts = np.bincount(val_metrics['pred_labels'], minlength=num_classes)
    payload = {
        'epoch': epoch,
        'lr': lr,
        'epoch_time_s': secs,
        'loss/train_ce': train_loss,
        'loss/val_ce': val_loss,
        'train_eval/qwk': train_metrics['qwk'],
        'train_eval/accuracy': train_metrics['accuracy'],
        'train_eval/auc_macro': train_metrics['auc_macro'],
        'val/qwk': val_metrics['qwk'],
        'val/accuracy': val_metrics['accuracy'],
        'val/loss': val_metrics['loss'],
        'val/auc_macro': val_metrics['auc_macro'],
        'val/kappa': val_metrics['kappa'],
    }
    for grade in range(num_classes):
        payload[f'val/recall_grade_{grade}'] = float(recall[grade])
        payload[f'val/pred_count_grade_{grade}'] = int(pred_counts[grade])
    table = wandb.Table(columns=['gt\\pred'] + [str(i) for i in range(num_classes)])
    for i in range(num_classes):
        table.add_data(str(i), *[int(v) for v in cm[i]])
    payload['val/confusion_matrix'] = table
    run.log(payload, step=epoch)


def save_checkpoint(ckpt_path, args, model, optimizer, scheduler, epoch,
                    best_metric, best_epoch, wandb_run_id=None):
    """Atomically save everything needed to resume as if the run never stopped."""
    model_to_use = _unwrap_model(model)
    state = {
        'epoch': epoch,  # last completed epoch
        'model': model_to_use.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'best_metric': best_metric,
        'best_epoch': best_epoch,
        'wandb_run_id': wandb_run_id,
        'config': {
            'model': 'SegRDNet', 'pooling': args.pooling, 'num_classes': args.num_classes,
            'img_size': args.img_size, 'optimizer': args.optimizer, 'lr': args.lr,
            'weight_decay': args.weight_decay, 'scheduler_name': args.scheduler,
            'label_smoothing': args.label_smoothing, 'batch_size': args.batch_size,
            'use_softcam_decoder0': args.use_softcam_decoder0,
        },
        'rng': {
            'torch': torch.get_rng_state(),
            'numpy': np.random.get_state(),
            'python': random.getstate(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    tmp_path = ckpt_path + '.tmp'
    torch.save(state, tmp_path)
    os.replace(tmp_path, ckpt_path)
    return ckpt_path


def load_checkpoint(path, model, optimizer, scheduler, device):
    """Restore model/optimizer/scheduler and RNG states; returns the raw state dict."""
    model_to_use = _unwrap_model(model)
    state = torch.load(path, map_location='cpu', weights_only=False)
    model_to_use.load_state_dict(state['model'])
    model_to_use.to(device)
    if optimizer is not None and state.get('optimizer'):
        optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None and state.get('scheduler'):
        scheduler.load_state_dict(state['scheduler'])
    rng = state.get('rng') or {}
    if rng.get('torch') is not None:
        torch.set_rng_state(rng['torch'])
    if rng.get('numpy') is not None:
        np.random.set_state(rng['numpy'])
    if rng.get('python') is not None:
        random.setstate(rng['python'])
    if rng.get('cuda') is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(rng['cuda'])
        except Exception as exc:
            print(f'Warning: could not restore CUDA RNG state: {exc}')
    return state


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
    parser.add_argument('--pooling', default='avg', choices=['lse', 'avg'], type=str,
                        help='Spatial pooling of the evidence maps: lse (original) or avg')
    
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
    parser.add_argument('--softcam-decoder0-mode', choices=['a', 'b', 'c'], default='a', help='A: head only; B: preserve features and refine evidence branch; C: replace Decoder 0 and project features to Decoder 1')
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
    
    # wandb / resume
    parser.add_argument('--wandb', action='store_true', help='Log per-epoch metrics to Weights & Biases')
    parser.add_argument('--wandb-project', default='MICCAI', type=str)
    parser.add_argument('--wandb-entity', default=None, type=str)
    parser.add_argument('--wandb-run-name', default=None, type=str)
    parser.add_argument('--resume', default=None, type=str,
                        help='Resume from a checkpoint: path to checkpoint.pt or its directory')
    
    # Debug / sanity modes
    parser.add_argument('--smoke', action='store_true', help='Train on 200 images, cap epochs at 2')
    parser.add_argument('--tiny-overfit', action='store_true',
                        help='Overfit 64 val-transform images for 50 optimizer steps; prints training accuracy')
    
    args = parser.parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 2)
    return args


def main():
    args = parse_args()
    cfg = Config(args)
    
    # Print config - NO driver detection, just is_available
    print("="*60)
    print("CHEN'S UNET (SegRDNet) APTOS TRAINING CONFIG")
    print("="*60)
    print(f"Device: {cfg.device}")
    print(f"Processes: {cfg.world_size} (rank {cfg.rank}); GPUs available: {cfg.num_gpus}")
    if cfg.world_size > 1:
        print("  DistributedDataParallel: one process per GPU (forward hooks are per-process, hook-safe)")
    elif cfg.num_gpus > 1:
        print("  Only one GPU in use; launch via torch.distributed.run to use all GPUs")
    print(f"Epochs: {cfg.epochs}")
    if cfg.smoke or cfg.tiny_overfit:
        modes = []
        if cfg.smoke:
            modes.append('smoke (train capped at 200 imgs, epochs capped at 2)')
        if cfg.tiny_overfit:
            modes.append('tiny-overfit (64 imgs, 50 steps)')
        print(f"Mode: {', '.join(modes)}")
    print(f"Seed: {args.seed}")
    print(f"Batch Size: {cfg.batch_size}/GPU (effective: {cfg.batch_size * cfg.grad_accum_steps * cfg.world_size} with grad_accum={cfg.grad_accum_steps}, world={cfg.world_size})")
    print(f"Workers: {cfg.num_workers} (persistent: {cfg.persistent_workers})")
    print(f"Loss: CrossEntropy (label_smoothing={cfg.label_smoothing})")
    
    # Option A: SoftCAM Decoder 0
    if cfg.use_softcam_decoder0:
        print(f"SoftCAM Decoder 0: ENABLED (Option {cfg.softcam_decoder0_mode.upper()})")
        print(f"  L1 lambda: {cfg.softcam_lambda_l1}")
        print(f"  L2 lambda: {cfg.softcam_lambda_l2}")
    else:
        print(f"SoftCAM Decoder 0: DISABLED (baseline)")
    
    print(f"Output: {cfg.save_path}")
    print(f"wandb: {'on, project ' + args.wandb_project + (f' (entity {args.wandb_entity})' if args.wandb_entity else '') if args.wandb else 'off'}")
    print(f"Resume: {args.resume or 'no (fresh run)'}")
    print("="*60)
    
    # Rank-offset seed for augmentation diversity; the split below uses an
    # explicit generator so it is identical on every rank
    set_random_seed(args.seed + cfg.rank)
    
    save_path = load_save_paths(cfg)
    if cfg.world_size > 1:
        # Each rank stamps its own timestamp; agree on rank 0's so all ranks
        # read/write the same run directory
        holder = [save_path]
        dist.broadcast_object_list(holder, src=0)
        save_path = holder[0]
        print(f'Save path (broadcast from rank 0): {save_path}')
    else:
        print(f'Save path: {save_path}')
    os.makedirs(save_path, exist_ok=True)
    
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
    
    full_dataset = APTOSDataset(train_csv_path, train_img_dir, transform=None, mode='train')
    split_rng = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(full_dataset), generator=split_rng).tolist()
    train_size = int(0.8 * len(full_dataset))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Separate dataset instances so val transforms never leak into training images
    train_source = APTOSDataset(train_csv_path, train_img_dir, transform=train_transform, mode='train')
    val_source = APTOSDataset(train_csv_path, train_img_dir, transform=val_transform, mode='val')
    if cfg.smoke and len(train_indices) > 200:
        train_indices = train_indices[:200]
        print('Smoke mode: training set capped at 200 images')
    train_dataset = Subset(train_source, train_indices)
    val_dataset = Subset(val_source, val_indices)
    
    print(f'Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}')
    
    # Shard the split across ranks when running DDP
    train_sampler = DistributedSampler(train_dataset, num_replicas=cfg.world_size, rank=cfg.rank,
                                       shuffle=True, seed=args.seed) if cfg.world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=cfg.world_size, rank=cfg.rank,
                                     shuffle=False) if cfg.world_size > 1 else None
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.num_workers, 
        pin_memory=True, 
        drop_last=True,
        persistent_workers=cfg.persistent_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.val_batch_size, 
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.num_workers, 
        pin_memory=True,
        persistent_workers=cfg.persistent_workers
    )
    
    # Model - evidence-map pooling selectable via --pooling
    model = SegRDNet(
        num_classes=cfg.num_classes,
        pooling=cfg.pooling,
        use_softcam_decoder0=cfg.use_softcam_decoder0,
        softcam_decoder0_mode=cfg.softcam_decoder0_mode,
        softcam_lambda_l1=cfg.softcam_lambda_l1,
        softcam_lambda_l2=cfg.softcam_lambda_l2
    )
    
    model.to(cfg.device)
    
    # SegRDNet reads intermediate features from self.activation, populated by
    # forward-hook closures bound to `self`. DataParallel replicates the module
    # per device but runs all replicas as threads in ONE process sharing this
    # object, so those writes race across GPUs. DDP gives each GPU its own
    # process and replica, keeping the hook payloads local and consistent.
    if cfg.world_size > 1:
        model = NativeDDP(model, device_ids=[cfg.device.index] if cfg.device.type == 'cuda' else None,
                          find_unused_parameters=True)
        if cfg.is_main:
            print(f"Wrapped model in DistributedDataParallel (world_size={cfg.world_size})")
    
    if cfg.is_main:
        base = _unwrap_model(model)
        total_params = sum(p.numel() for p in base.parameters())
        trainable_params = sum(p.numel() for p in base.parameters() if p.requires_grad)
        print(f'Model: SegRDNet, Parameters: {total_params:,} | trainable: {trainable_params:,}')
        print(pretrained_weights_status(base, cfg.num_classes))
    
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
    
    # Tiny-overfit sanity mode: overfit 64 images, report train accuracy, exit
    if cfg.tiny_overfit:
        tiny_overfit(cfg, model, optimizer, loss_fn, amp_autocast, loss_scaler)
        logger.close()
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        print('\nDone (tiny-overfit mode; skipping epoch training, checkpointing and submission).')
        return 0
    
    # Resume: restore weights, optimizer, scheduler, best tracking and RNG states
    resume_state = None
    start_epoch = 0
    if args.resume:
        resume_path = args.resume
        if os.path.isdir(resume_path):
            resume_path = os.path.join(resume_path, 'checkpoint.pt')
        if not os.path.isfile(resume_path):
            raise SystemExit(f'Resume checkpoint not found: {resume_path}')
        resume_state = load_checkpoint(resume_path, model, optimizer, scheduler, cfg.device)
        start_epoch = resume_state['epoch'] + 1
        best_metric = resume_state.get('best_metric', float('-inf'))
        best_epoch = resume_state.get('best_epoch', 0)
        if resume_state.get('config') and resume_state['config'].get('use_softcam_decoder0') != cfg.use_softcam_decoder0:
            print(f"WARNING: checkpoint was trained with use_softcam_decoder0="
                  f"{resume_state['config'].get('use_softcam_decoder0')}, "
                  f"current value is {cfg.use_softcam_decoder0}")
        print(f'Resumed from {resume_path}: continuing at epoch {start_epoch} '
              f'(best QWK so far {best_metric:.4f} at epoch {best_epoch}). '
              f'Use the same --seed to reproduce the split.')
    else:
        # -inf so the first epoch always snapshots best_model.pth, even when
        # early QWK is 0 or negative (untrained models can predict one class)
        best_metric = float('-inf')
        best_epoch = 0

    run = init_wandb(args, resume_state=resume_state) if cfg.is_main else None
    wandb_run_id = run.id if run is not None else None

    for epoch in range(start_epoch, cfg.epochs):
        epoch_start = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
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
        
        # Save best model (by QWK); metrics are global, so all ranks agree here
        metric = val_metrics['qwk']
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            if cfg.is_main:
                torch.save(_unwrap_model(model).state_dict(), os.path.join(save_path, 'best_model.pth'))
                print(f'  -> Best model saved (QWK: {best_metric:.4f})')
        
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metric)
        
        # Rolling resume checkpoint + wandb metrics for this epoch (rank 0 only)
        lr_now = optimizer.param_groups[0]['lr']
        secs = time.time() - epoch_start
        if cfg.is_main:
            save_checkpoint(
                os.path.join(save_path, 'checkpoint.pt'), args, model, optimizer, scheduler,
                epoch, best_metric, best_epoch, wandb_run_id=wandb_run_id,
            )
            wandb_log_epoch(
                run, epoch, lr_now, secs, train_metrics['loss'], val_metrics['loss'],
                train_metrics, val_metrics, cfg.num_classes,
            )
    
    if run is not None:
        run.finish()
    
    # All ranks are done training: sync once, then tear down the process group
    # before the rank-0 final evaluation (no collectives may run after this)
    is_distributed = dist.is_available() and dist.is_initialized()
    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()
    
    if cfg.is_main:
        print(f'\n{"="*60}')
        print(f'Best validation QWK: {best_metric:.4f} at epoch {best_epoch}')
        print(f'{"="*60}')
        
        # Full (non-sharded) val loader so final metrics cover the whole split
        if is_distributed:
            val_loader = DataLoader(
                val_dataset,
                batch_size=cfg.val_batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=True,
                persistent_workers=cfg.persistent_workers,
            )
        
        # Final evaluation with best model (fall back to current weights if
        # best_model.pth was never written, e.g. QWK stayed <= 0 every epoch)
        print('\n=== Final Evaluation (Best Model) ===')
        best_model_path = os.path.join(save_path, 'best_model.pth')
        if os.path.isfile(best_model_path):
            best_state = torch.load(best_model_path, map_location=cfg.device)
            _unwrap_model(model).load_state_dict(best_state)
        else:
            print('WARNING: best_model.pth was never saved (validation QWK never '
                  'exceeded the initial best); evaluating the current weights instead.')
        final_metrics = validate(model, val_loader, loss_fn, cfg, cfg.device, amp_autocast)
        print(f"FINAL: QWK={final_metrics['qwk']:.4f}, Acc={final_metrics['accuracy']:.4f}, AUC={final_metrics['auc_macro']:.4f}")
        print(f"Confusion Matrix:\n{final_metrics['confusion_matrix']}")
        
        # Generate submission
        print('\nGenerating submission...')
        generate_submission(cfg, model, save_path, val_transform)
        
        logger.close()
        if is_distributed:
            print('\nDone! (Distributed: rank 0 wrote all outputs; other ranks exited earlier.)')
        else:
            print('\nDone!')


if __name__ == '__main__':
    main()
