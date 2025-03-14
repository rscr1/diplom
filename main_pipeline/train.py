"""
Training pipeline for depth estimation and segmentation models.
This module implements training functionality for various depth estimation and segmentation architectures.
"""
# Standard library imports
import os
from collections import OrderedDict
from typing import Optional

# Third-party imports
from clearml import Task
import segmentation_models_pytorch as smp
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex, Dice

# Local application imports
from augment import augment
from config import Config
from dataset import CustomDepthDataset, CustomSegmDataset
from metrics.mse import MeanAbsoluteError, MeanSquaredError
from multihead_model import *
from train_loop import train_loop
from utils_func import set_seed, find_path, log_config_to_clearml, gradient_off
from Loss import MSELoss, Loss_function

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
def run(config: Config) -> None:
    """
    Run the training pipeline for depth estimation and segmentation models.

    Args:
        config: Complete configuration object
        
    Returns:
        None
    """
    set_seed(config.training.seed)    

    path, launch = find_path(config.output.save_dir)
    print(f'Results will be saved to {path}')
    
    checkpoints_path = f'{path}/weights'
    os.makedirs(checkpoints_path, exist_ok=True)
    metrics_path = f'{path}/metrics.png'
    losses_path = f'{path}/losses.png'
    dataframe_path = f'{path}/frame.csv'

    is_segm_task = config.task.type == 'segm'

    # Initialize or get ClearML task
    task = Task.init(
        project_name=config.output.project_name,
        task_name=f'launch_{launch}',
        tags=[
            config.task.type,
            f'{config.data.image_size.width}x{config.data.image_size.height}',
            config.model.name,
            config.model.encoder,
            config.training.optimizer.name
        ],
        reuse_last_task_id=False
    )
    
    # Log configuration
    log_config_to_clearml(task, config)

    # Setup data transforms and loaders
    train_transforms, valid_transforms = augment(
        config=config.data.augmentation,
        height=config.data.image_size.height,
        width=config.data.image_size.width
    )

    # Initialize appropriate dataset based on task
    if is_segm_task:
        train_dataset = CustomSegmDataset(config.data.train_path, train_transforms)
        valid_dataset = CustomSegmDataset(config.data.valid_path, valid_transforms)
    else:
        train_dataset = CustomDepthDataset(config.data.train_path, train_transforms)
        valid_dataset = CustomDepthDataset(config.data.valid_path, valid_transforms)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        drop_last=True
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        drop_last=True
    )

    dataloader = {
        'train': train_dataloader,
        'valid': valid_dataloader
    }

    # Model initialization
    if config.model.name == 'deeplabv3':
        model = smp.DeepLabV3(
            encoder_name=config.model.encoder,
            classes=config.task.classes,
            encoder_weights="imagenet",
            upsampling=32
        )
        if config.model.weights is not None:
            model.load_state_dict(torch.load(config.model.weights, map_location=torch.device('cpu')))
    elif config.model.name == 'fpn':
        model = smp.FPN(
            encoder_name=config.model.encoder,
            classes=config.task.classes,
            encoder_weights="imagenet"
        )
        if config.model.weights is not None:
            model.load_state_dict(torch.load(config.model.weights, map_location=torch.device('cpu')))
    elif config.model.name == 'deeplabv3+':
        model = smp.DeepLabV3(
            encoder_name=config.model.encoder,
            classes=config.task.classes,
            encoder_weights="imagenet"
        )
        if config.model.weights is not None:
            model.load_state_dict(torch.load(config.model.weights, map_location=torch.device('cpu')))
    elif config.model.name == 'segm_fpn_depth_head':
        model = SegmFPNWithDepthHead(
            encoder_name=config.model.encoder,
            classes=config.task.classes,
            encoder_weights="imagenet"
        )
        model.load_state_dict(torch.load(config.model.weights), strict=False)
        gradient_off(model, is_segm_task)
    elif config.model.name == 'segm_fpn_depth_decoder_head':
        model = SegmFPNWithDepthDecoderHead(
            encoder_name=config.model.encoder,
            classes=config.task.classes,
            encoder_weights="imagenet"
        )
        model.load_state_dict(torch.load(config.model.weights), strict=False)
        gradient_off(model, is_segm_task)
    elif config.model.name == 'segm_deeplab_depth_head':
        model = SegmDeepLabV3WithDepthHead(
            encoder_name=config.model.encoder,
            classes=config.task.classes,
            encoder_weights="imagenet"
        )
        model.load_state_dict(torch.load(config.model.weights), strict=False)
        gradient_off(model, is_segm_task)
    elif config.model.name == 'segm_deeplab_depth_decoder_head':
        model = SegmDeepLabV3WithDepthDecoderHead(
            encoder_name=config.model.encoder,
            classes=config.task.classes,
            encoder_weights="imagenet"
        )
        model.load_state_dict(torch.load(config.model.weights), strict=False)
        gradient_off(model, is_segm_task)
    elif config.model.name == 'depth_fpn_segm_head':
        model = DepthFPNWithSegmHead(
            encoder_name=config.model.encoder,
            classes=config.task.classes,
            encoder_weights="imagenet"
        )
        state_dict = torch.load(config.model.weights)
        state_dict = OrderedDict([('depth_head', v) if k == 'segmentation_head' else (k, v) for k, v in state_dict.items()])
        model.load_state_dict(state_dict, strict=False)
        gradient_off(model, is_segm_task)
    elif config.model.name == 'depth_fpn_segm_decoder_head':
        model = DepthFPNWithSegmDecoderHead(
            encoder_name=config.model.encoder,
            classes=config.task.classes,
            encoder_weights="imagenet"
        )
        state_dict = torch.load(config.model.weights)
        state_dict = OrderedDict([('depth_head', v) if k == 'segmentation_head' else (k, v) for k, v in state_dict.items()])
        model.load_state_dict(state_dict, strict=False)
        gradient_off(model, is_segm_task)
    elif config.model.name == 'depth_deeplab_segm_head':
        model = DepthDeepLabV3WithSegmHead(
            encoder_name=config.model.encoder,
            classes=config.task.classes,
            encoder_weights="imagenet"
        )
        state_dict = torch.load(config.model.weights)
        state_dict = OrderedDict([('depth_head', v) if k == 'segmentation_head' else (k, v) for k, v in state_dict.items()])
        model.load_state_dict(state_dict, strict=False)
        gradient_off(model, is_segm_task)
    elif config.model.name == 'depth_deeplab_segm_decoder_head':
        model = DepthDeepLabV3WithSegmDecoderHead(
            encoder_name=config.model.encoder,
            classes=config.task.classes,
            encoder_weights="imagenet"
        )
        state_dict = torch.load(config.model.weights)
        state_dict = OrderedDict([('depth_head', v) if k == 'segmentation_head' else (k, v) for k, v in state_dict.items()])
        model.load_state_dict(state_dict, strict=False)
        gradient_off(model, is_segm_task)

    # Device setup and model parallelization
    if config.hardware.device == 'cpu':
        model = model.to(config.hardware.device)
    else:
        if config.hardware.parallel.type == 'ddp':
            model = nn.parallel.DistributedDataParallel(model)
        elif config.hardware.parallel.type == 'dp':
            model = nn.DataParallel(model, device_ids=config.hardware.parallel.device_ids)
            device = f'cuda:{config.hardware.parallel.device_ids[0]}'
            model = model.to(device)
        else:
            model = model.to(device)

    print(f'Config/Available count of devices: {len(config.hardware.parallel.device_ids)}/{torch.cuda.device_count()}')
    print(f'Main device: {device}')

    # Initialize metrics
    metrics = {}
    if is_segm_task:  # Segmentation metrics
        for metric_name in config.training.metrics.segmentation:
            if metric_name == 'iou':
                metrics['iou'] = JaccardIndex(
                    num_classes=config.task.classes,
                    average='none',
                    ignore_index=config.training.loss.ignore_index
                ).to(config.hardware.device)
            elif metric_name == 'dice':
                metrics['dice'] = Dice(
                    num_classes=config.task.classes,
                    average='none',
                    ignore_index=config.training.loss.ignore_index
                ).to(config.hardware.device)
    else:  # Depth metrics
        for metric_name in config.training.metrics.depth:
            if metric_name == 'mae':
                metrics['mae'] = MeanAbsoluteError().to(config.hardware.device)
            elif metric_name == 'mse':
                metrics['mse'] = MeanSquaredError().to(config.hardware.device)
            elif metric_name == 'rmse':
                metrics['rmse'] = MeanSquaredError(squared=False).to(config.hardware.device)

    # Initialize loss function
    if is_segm_task:
        if isinstance(config.training.loss.name, str):
            if config.training.loss.name == 'focal':
                loss_func = smp.losses.FocalLoss(mode=config.task.mode, ignore_index=config.training.loss.ignore_index)
            elif config.training.loss.name == 'crossentropy':
                if config.model.loss_weights is not None and not torch.is_tensor(config.model.loss_weights):
                    config.model.loss_weights = torch.tensor(config.model.loss_weights, dtype=torch.float, device=config.hardware.device)
                loss_func = nn.CrossEntropyLoss(weight=config.model.loss_weights, ignore_index=config.training.loss.ignore_index)
            elif config.training.loss.name == 'lovasz':
                loss_func = smp.losses.LovaszLoss(mode=config.task.mode, ignore_index=config.training.loss.ignore_index)
            elif config.training.loss.name == 'dice':
                loss_func = smp.losses.DiceLoss(mode=config.task.mode, ignore_index=config.training.loss.ignore_index)
            elif config.training.loss.name == 'jaccard':
                loss_func = smp.losses.JaccardLoss(mode=config.task.mode, ignore_index=config.training.loss.ignore_index)
            elif config.training.loss.name == 'tversky':
                loss_func = smp.losses.TverskyLoss(mode=config.task.mode, ignore_index=config.training.loss.ignore_index)
        else:
            loss_func = Loss_function(mode=config.task.mode, losses=config.training.loss.name, device=config.hardware.device)
    else:
        if config.training.loss.name == 'mse':
            loss_func = MSELoss(reduction='sum')

    # Initialize optimizer
    if model is not None:
        if config.training.optimizer.name == 'Adam':
            optimizer = torch.optim.Adam(
                params=model.parameters(),
                lr=config.training.optimizer.lr
            )
        elif config.training.optimizer.name == 'AdamW':
            optimizer = torch.optim.AdamW(
                params=model.parameters(),
                lr=config.training.optimizer.lr
            )
        elif config.training.optimizer.name == 'SGD':
            optimizer = torch.optim.SGD(
                params=model.parameters(),
                lr=config.training.optimizer.lr
            )

    # Initialize scheduler
    scheduler = None
    if optimizer is not None:
        if config.training.scheduler.name == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.training.scheduler.step_size
            )
        elif config.training.scheduler.name == 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=config.training.scheduler.max_lr,
                epochs=config.training.epochs,
                steps_per_epoch=len(train_dataloader)
            )
        elif config.training.scheduler.name == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=config.training.scheduler.mode,
                factor=config.training.scheduler.factor,
                patience=config.training.scheduler.patience
            )
        
    # Training loop
    train_loop(
        model=model,
        dataloader=dataloader,
        loss_func=loss_func,
        optimizer=optimizer,
        metrics=metrics,
        parallel_type=config.hardware.parallel.type,
        device=config.hardware.device,
        classes=config.task.classes,
        early_stopping=config.training.early_stopping,
        is_segm_task=is_segm_task,
        model_name=config.model.name,
        scheduler=scheduler,
        epochs=config.training.epochs,
        checkpoints_path=checkpoints_path,
        losses_path=losses_path,
        metrics_path=metrics_path,
        dataframe_path=dataframe_path
    )
