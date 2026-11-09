#!/usr/bin/env python3
"""
Kaggle Notebook: SegRDNet for APTOS 2019 Blindness Detection
5-class Diabetic Retinopathy Classification
Repository: Beyond-Classification-Elaborating-Network-Predictions-for-Better-Weakly-Supervised-Quantization
"""

# =============================================================================
# INSTALL DEPENDENCIES (run once in Kaggle)
# =============================================================================
# !pip install -q timm pytorch-grad-cam albumentations torchmetrics

# =============================================================================
# IMPORTS
# =============================================================================
import os
import sys
import time
import yaml
import json
import argparse
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from collections import OrderedDict
from contextlib import suppress
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision.transforms as T
from PIL import Image

from timm import utils
from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.layers import convert_splitbn_model, convert_sync_batchnorm, set_fast_norm
from timm.loss import JsdCrossEntropy, SoftTargetCrossEntropy, BinaryCrossEntropy, LabelSmoothingCrossEntropy
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, model_parameters
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.utils import ApexScaler, NativeScaler

from torchmetrics.functional.classification import multiclass_auroc, multiclass_accuracy

# Add the model path
sys.path.append('/kaggle/working/Beyond-Classification-Elaborating-Network-Predictions-for-Better-Weakly-Supervised-Quantization')
from models.model_segrdnet import SegRDNet
from models.rdnet import rdnet_tiny

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

def get_aptos_transforms(img_size=512, mode='train'):
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
# ARGUMENT PARSER (adapted from train_stage_1.py)
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='SegRDNet APTOS Training')
    
    # Data
    parser.add_argument('--data-dir', default='/kaggle/input/competitions/aptos2019-blindness-detection', type=str)
    parser.add_argument('--train-csv', default='train.csv', type=str)
    parser.add_argument('--test-csv', default='test.csv', type=str)
    parser.add_argument('--train-img-dir', default='train_images', type=str)
    parser.add_argument('--test-img-dir', default='test_images', type=str)
    parser.add_argument('--img-size', default=512, type=int)
    
    # Model
    parser.add_argument('--model', default='rdnet_tiny', type=str)
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--num-classes', default=5, type=int)
    parser.add_argument('--initial-checkpoint', default='', type=str)
    parser.add_argument('--resume', default='', type=str)
    
    # Training
    parser.add_argument('--batch-size', default=16, type=int)  # Reduced for Kaggle GPU memory
    parser.add_argument('--validation-batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--lr', default=None, type=float)
    parser.add_argument('--lr-base', default=1e-3, type=float)
    parser.add_argument('--lr-base-size', default=256, type=int)
    parser.add_argument('--sched', default='cosine', type=str)
    parser.add_argument('--warmup-epochs', default=5, type=int)
    parser.add_argument('--min-lr', default=1e-6, type=float)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--opt', default='adamw', type=str)
    parser.add_argument('--drop', default=0.0, type=float)
    parser.add_argument('--drop-path', default=0.1, type=float)
    parser.add_argument('--smoothing', default=0.1, type=float)
    parser.add_argument('--bce-loss', action='store_true', default=False)  # Use CE for multi-class
    parser.add_argument('--mixup', default=0.5, type=float)
    parser.add_argument('--cutmix', default=1.0, type=float)
    parser.add_argument('--mixup-prob', default=1.0, type=float)
    parser.add_argument('--mixup-switch-prob', default=0.5, type=float)
    parser.add_argument('--mixup-mode', default='batch', type=str)
    
    # System
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--amp', action='store_true', default=True)
    parser.add_argument('--amp-dtype', default='float16', type=str)
    parser.add_argument('--output', default='/kaggle/working/output', type=str)
    parser.add_argument('--experiment', default='segrdnet_aptos', type=str)
    parser.add_argument('--eval-metric', default='auc_mean', type=str)
    parser.add_argument('--log-interval', default=50, type=int)
    parser.add_argument('--checkpoint-hist', default=5, type=int)
    parser.add_argument('--log-wandb', action='store_true', default=False)
    
    # Distributed (not used in Kaggle single GPU)
    parser.add_argument('--local-rank', default=0, type=int)
    parser.add_argument('--distributed', action='store_true', default=False)
    
    args = parser.parse_args([])
    return args

# =============================================================================
# TRAINING FUNCTIONS (adapted from train_stage_1.py)
# =============================================================================
_logger = logging.getLogger('train')

def train_one_epoch(epoch, model, loader, optimizer, loss_fn, args, device, 
                    lr_scheduler=None, saver=None, output_dir=None, 
                    amp_autocast=suppress, loss_scaler=None, model_ema=None, mixup_fn=None):
    
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    has_no_sync = hasattr(model, "no_sync")
    all_predictions = []
    all_targets = []

    update_time_m = utils.AverageMeter()
    data_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()

    model.train()

    accum_steps = args.grad_accum_steps
    last_accum_steps = len(loader) % accum_steps
    updates_per_epoch = (len(loader) + accum_steps - 1) // accum_steps
    num_updates = epoch * updates_per_epoch
    last_batch_idx = len(loader) - 1
    last_batch_idx_to_accum = len(loader) - last_accum_steps

    data_start_time = update_start_time = time.time()
    optimizer.zero_grad()
    update_sample_count = 0
    
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_batch_idx
        need_update = last_batch or (batch_idx + 1) % accum_steps == 0
        update_idx = batch_idx // accum_steps
        if batch_idx >= last_batch_idx_to_accum:
            accum_steps = last_accum_steps
        
        if not args.prefetcher:
            input, target = input.to(device), target.to(device)
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        # For multi-class classification, target stays as long tensor
        data_time_m.update(accum_steps * (time.time() - data_start_time))

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
                           create_graph=second_order, need_update=need_update)
            else:
                _loss.backward(create_graph=second_order)
                if need_update:
                    if args.clip_grad is not None:
                        utils.dispatch_clip_grad(
                            model_parameters(model, exclude_head='agc' in args.clip_mode),
                            value=args.clip_grad, mode=args.clip_mode)
                    optimizer.step()

        if has_no_sync and not need_update:
            with model.no_sync():
                loss, output = _forward()
                all_predictions += [output]
                all_targets += [target]
                _backward(loss)
        else:
            loss, output = _forward()
            all_predictions += [output]
            all_targets += [target]
            _backward(loss)

        if not args.distributed:
            losses_m.update(loss.item() * accum_steps, input.size(0))
        update_sample_count += input.size(0)

        if not need_update:
            data_start_time = time.time()
            continue

        num_updates += 1
        optimizer.zero_grad()
        if model_ema is not None:
            model_ema.update(model)

        if args.synchronize_step and device.type == 'cuda':
            torch.cuda.synchronize()
        time_now = time.time()
        update_time_m.update(time_now - update_start_time)
        update_start_time = time_now

        if update_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            if utils.is_primary(args):
                _logger.info(f'Train: {epoch} [{update_idx:>4d}/{updates_per_epoch} '
                           f'({100. * update_idx / (updates_per_epoch - 1):>3.0f}%)]  '
                           f'Loss: {losses_m.val:#.3g} ({losses_m.avg:#.3g})  '
                           f'Time: {update_time_m.val:.3f}s  '
                           f'LR: {lr:.3e}  '
                           f'Data: {data_time_m.val:.3f}')

        if saver is not None and args.recovery_interval and (update_idx + 1) % args.recovery_interval == 0:
            saver.save_recovery(epoch, batch_idx=update_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        update_sample_count = 0
        data_start_time = time.time()

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    p_test = torch.cat(all_predictions, dim=0)
    y_test = torch.cat(all_targets, dim=0)
    
    # Multi-class AUC
    auroc_score = multiclass_auroc(p_test, y_test, num_classes=args.num_classes, average=None).cpu().numpy()
    acc_score = multiclass_accuracy(p_test, y_test, num_classes=args.num_classes, average='macro').cpu().numpy()
    print(f'train auc: {auroc_score}, macro: {auroc_score.mean():.4f}')
    print(f'train acc: {acc_score:.4f}')
    
    metrics = OrderedDict([('loss', losses_m.avg), ('auc', auroc_score), ('auc_mean', auroc_score.mean()), ('acc', acc_score)])
    return metrics


def validate(model, loader, loss_fn, args, device, amp_autocast=suppress, log_suffix=''):
    batch_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    all_predictions = []
    all_targets = []

    model.eval()
    end = time.time()
    last_idx = len(loader) - 1
    
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.to(device)
                target = target.to(device)
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)
                if isinstance(output, (tuple, list)):
                    output = output[0]
                reduce_factor = args.tta
                if reduce_factor > 1:
                    output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                    target = target[0:target.size(0):reduce_factor]
                loss = loss_fn(output, target)

            all_predictions += [output]
            all_targets += [target]
            
            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
            else:
                reduced_loss = loss.data

            if device.type == 'cuda':
                torch.cuda.synchronize()
            losses_m.update(reduced_loss.item(), input.size(0))
            batch_time_m.update(time.time() - end)
            end = time.time()
            
            if utils.is_primary(args) and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(f'{log_name}: [{batch_idx:>4d}/{last_idx}]  '
                           f'Time: {batch_time_m.val:.3f} ({batch_time_m.avg:.3f})  '
                           f'Loss: {losses_m.val:>7.3f} ({losses_m.avg:>6.3f})')

    p_test = torch.cat(all_predictions, dim=0)
    y_test = torch.cat(all_targets, dim=0)
    auroc_score = multiclass_auroc(p_test, y_test, num_classes=args.num_classes, average=None).cpu().numpy()
    acc_score = multiclass_accuracy(p_test, y_test, num_classes=args.num_classes, average='macro').cpu().numpy()
    print(f'val auc: {auroc_score}, macro: {auroc_score.mean():.4f}')
    print(f'val acc: {acc_score:.4f}')
    
    metrics = OrderedDict([('loss', losses_m.avg), ('auc', auroc_score), ('auc_mean', auroc_score.mean()), ('acc', acc_score)])
    return metrics


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================
def main():
    args = parse_args()
    utils.setup_default_logging()
    
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    args.prefetcher = not args.no_prefetcher if hasattr(args, 'no_prefetcher') else True
    args.grad_accum_steps = max(1, getattr(args, 'grad_accum_steps', 1))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    args.rank = 0
    args.world_size = 1
    args.distributed = False

    # AMP setup
    use_amp = args.amp
    amp_dtype = torch.float16 if args.amp_dtype == 'float16' else torch.bfloat16
    has_native_amp = hasattr(torch.cuda.amp, 'autocast')
    
    if use_amp and not has_native_amp:
        print("Native AMP not available, disabling AMP")
        use_amp = False

    utils.random_seed(args.seed, args.rank)
    
    if args.fuser if hasattr(args, 'fuser') else False:
        utils.set_jit_fuser(args.fuser)
    if getattr(args, 'fast_norm', False):
        set_fast_norm()

    # Create model
    print(f'Creating model: {args.model}')
    model = SegRDNet(num_classes=args.num_classes)
    
    if args.initial_checkpoint:
        print(f'Loading initial checkpoint: {args.initial_checkpoint}')
        checkpoint = torch.load(args.initial_checkpoint, map_location='cpu')
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)

    data_config = resolve_data_config(vars(args), model=model, verbose=True)
    
    num_aug_splits = 0
    if getattr(args, 'aug_splits', 0) > 0:
        num_aug_splits = args.aug_splits

    if getattr(args, 'split_bn', False):
        assert num_aug_splits > 1 or getattr(args, 'resplit', False)
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    model.to(device=device)
    if getattr(args, 'channels_last', False):
        model.to(memory_format=torch.channels_last)

    # AMP autocast
    if use_amp:
        amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
        loss_scaler = NativeScaler()
        print('Using native Torch AMP. Training in mixed precision.')
    else:
        amp_autocast = suppress
        loss_scaler = None
        print('AMP not enabled. Training in float32.')

    # Resume
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(model, args.resume, optimizer=None, 
                                         loss_scaler=None, log_info=True)

    # EMA
    model_ema = None
    if getattr(args, 'model_ema', False):
        model_ema = utils.ModelEmaV2(model, decay=args.model_ema_decay, 
                                     device='cpu' if args.model_ema_force_cpu else None)
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    # DataLoaders
    train_transform = get_aptos_transforms(args.img_size, 'train')
    val_transform = get_aptos_transforms(args.img_size, 'val')
    
    train_csv_path = os.path.join(args.data_dir, args.train_csv)
    test_csv_path = os.path.join(args.data_dir, args.test_csv)
    train_img_dir = os.path.join(args.data_dir, args.train_img_dir)
    test_img_dir = os.path.join(args.data_dir, args.test_img_dir)
    
    # Split train into train/val (80/20)
    full_dataset = APTOSDataset(train_csv_path, train_img_dir, transform=train_transform, mode='train')
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transform
    val_dataset.dataset.mode = 'val'
    
    print(f'Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}')

    # Mixup
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0 or getattr(args, 'cutmix_minmax', None) is not None
    if mixup_active:
        mixup_args = dict(mixup_alpha=args.mixup, cutmix_alpha=args.cutmix,
                         cutmix_minmax=getattr(args, 'cutmix_minmax', None),
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
    loader_eval = DataLoader(val_dataset, batch_size=args.validation_batch_size or args.batch_size, 
                             shuffle=False, num_workers=args.workers, drop_last=False, pin_memory=True)

    # Loss
    if args.bce_loss:
        train_loss_fn = nn.BCEWithLogitsLoss()
    elif mixup_active:
        train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    train_loss_fn = train_loss_fn.to(device=device)
    validate_loss_fn = nn.CrossEntropyLoss().to(device=device)

    # Optimizer & Scheduler
    if not args.lr:
        global_batch_size = args.batch_size * args.world_size * args.grad_accum_steps
        batch_ratio = global_batch_size / args.lr_base_size
        args.lr_base_scale = getattr(args, 'lr_base_scale', '') or ('sqrt' if any(o in args.opt.lower() for o in ('ada', 'lamb')) else 'linear')
        if args.lr_base_scale == 'sqrt':
            batch_ratio = batch_ratio ** 0.5
        args.lr = args.lr_base * batch_ratio
        print(f'Learning rate: {args.lr}')

    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args), **getattr(args, 'opt_kwargs', {}))
    
    updates_per_epoch = (len(loader_train) + args.grad_accum_steps - 1) // args.grad_accum_steps
    lr_scheduler, num_epochs = create_scheduler_v2(optimizer, **scheduler_kwargs(args), 
                                                    updates_per_epoch=updates_per_epoch)
    
    start_epoch = 0
    if args.start_epoch is not None:
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    # Output dir
    if args.experiment:
        exp_name = args.experiment
    else:
        exp_name = '-'.join([datetime.now().strftime("%Y%m%d-%H%M%S"), 
                            safe_model_name(args.model), str(data_config['input_size'][-1])])
    output_dir = utils.get_outdir(args.output, exp_name)
    decreasing = True if args.eval_metric == 'loss' else False
    saver = utils.CheckpointSaver(model=model, optimizer=optimizer, args=args, model_ema=model_ema,
                                  amp_scaler=loss_scaler, checkpoint_dir=output_dir, 
                                  recovery_dir=output_dir, decreasing=decreasing, 
                                  max_history=args.checkpoint_hist)

    # Wandb
    if args.log_wandb:
        try:
            import wandb
            wandb.init(project=args.experiment, config=args)
        except ImportError:
            print("wandb not installed, skipping")

    print(f'Scheduled epochs: {num_epochs}. LR stepped per {"epoch" if lr_scheduler.t_in_epochs else "update"}.')
    
    try:
        for epoch in range(start_epoch, num_epochs):
            if hasattr(train_dataset, 'set_epoch'):
                train_dataset.set_epoch(epoch)
            
            train_metrics = train_one_epoch(epoch, model, loader_train, optimizer, train_loss_fn, args, device,
                                          lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                                          amp_autocast=amp_autocast, loss_scaler=loss_scaler, 
                                          model_ema=model_ema, mixup_fn=mixup_fn)
            
            eval_metrics = validate(model, loader_eval, validate_loss_fn, args, device, 
                                   amp_autocast=amp_autocast)
            
            if model_ema is not None and not args.model_ema_force_cpu:
                ema_eval_metrics = validate(model_ema.module, loader_eval, validate_loss_fn, args, device,
                                           amp_autocast=amp_autocast, log_suffix=' (EMA)')
                eval_metrics = ema_eval_metrics

            if output_dir is not None:
                lrs = [param_group['lr'] for param_group in optimizer.param_groups]
                utils.update_summary(epoch, train_metrics, eval_metrics, 
                                    filename=os.path.join(output_dir, 'summary.csv'),
                                    lr=sum(lrs) / len(lrs), write_header=(epoch==0),
                                    log_wandb=args.log_wandb)

            if saver is not None:
                save_metric = eval_metrics[args.eval_metric]
                saver.save_checkpoint(epoch, metric=save_metric)

            if lr_scheduler is not None:
                lr_scheduler.step(epoch + 1, eval_metrics[args.eval_metric])

    except KeyboardInterrupt:
        pass

    # Generate submission
    print("\nGenerating submission...")
    generate_submission(model, test_csv_path, test_img_dir, val_transform, args, device, output_dir)

    if args.log_wandb:
        try:
            import wandb
            wandb.finish()
        except:
            pass


def generate_submission(model, test_csv_path, test_img_dir, transform, args, device, output_dir):
    """Generate submission.csv for Kaggle"""
    model.eval()
    test_df = pd.read_csv(test_csv_path)
    test_dataset = APTOSDataset(test_csv_path, test_img_dir, transform=transform, mode='val')
    test_loader = DataLoader(test_dataset, batch_size=args.validation_batch_size or args.batch_size,
                             shuffle=False, num_workers=args.workers, pin_memory=True)
    
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
    print(f'Submission saved to {submission_path}')
    print(submission['diagnosis'].value_counts().sort_index())


if __name__ == '__main__':
    main()