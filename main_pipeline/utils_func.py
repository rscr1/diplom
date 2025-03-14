import os
import random

from clearml import Task
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

from config import Config


def set_seed(seed):
    np.random.seed(seed)
    np.random.RandomState(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_path(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)

    launch = 0
    path = f'{dir}/launch_{launch}'
    while os.path.exists(path):
        launch += 1
        path = f'{dir}/launch_{launch}'
    os.makedirs(path, exist_ok=True)
    return path, launch


def visualize(results_path, task='depth', **images):
    """Plot images based on task type.
    
    Args:
        results_path: Path to save the visualization
        task: Type of visualization ('depth', 'segm', 'both')
        **images: Dictionary of images to visualize
    """
    plt.figure(figsize=(16, 5))
    
    if task == 'depth':
        n = len(images)
        for i, (name, image) in enumerate(images.items()):
            plt.subplot(n, 1, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.title(' '.join(name.split('_')).title())
            if 'depth' in name.lower():
                plt.imshow(image)
            else:
                plt.imshow(image, cmap='inferno')
                plt.colorbar()
    
    elif task == 'segm':
        n = len(images)
        for i, (name, image) in enumerate(images.items()):
            plt.subplot(n, 1, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.title(' '.join(name.split('_')).title())
            plt.imshow(image)
    
    else:  # both
        n = len(images) // 2  # Предполагаем, что у нас две группы изображений
        for i, (name, image) in enumerate(images.items()):
            plt.subplot(2, n, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.title(' '.join(name.split('_')).title())
            if 'depth' in name.lower():
                plt.imshow(image, cmap='inferno')
                plt.colorbar()
            else:
                plt.imshow(image)

    plt.tight_layout()
    plt.savefig(results_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def rgb2classes(rgb_mask, colors):
    classes_mask = np.zeros((rgb_mask.shape[0], rgb_mask.shape[1]))
    for class_id, color in enumerate(colors):
        mask = np.all(rgb_mask == color, axis=-1)
        classes_mask[mask] = class_id
    print(classes_mask.shape)
    return classes_mask


def classes2rgb(classes_mask, colors):
    rgb_mask = np.zeros((classes_mask.shape[0], classes_mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(colors):
        mask = classes_mask == class_id
        rgb_mask[mask, :] = color
    return rgb_mask


def get_mask(out):
    if isinstance(out, tuple):
        _, mask = out
    else:
        mask = out

    return mask


def get_depth(out):
    if isinstance(out, tuple):
        depth, _ = out
    else:
        depth = out

    return depth


def bn_off(model, phase, is_segm_task, model_name):
    if phase == 'train':
        if model_name in {'fpn', 'deeplabv3', 'deeplabv3+'}:
            model.train()
        else:
            for name, module in model.named_modules():
                if ((is_segm_task and 'segmentation' not in name) or (not is_segm_task and 'depth' not in name)) and isinstance(module, nn.BatchNorm2d):
                    if hasattr(module, 'weight'):
                        module.weight.requires_grad_(False)
                    if hasattr(module, 'bias'):
                        module.bias.requires_grad_(False)
                    module.eval()
    else:
        model.eval()


def gradient_off(model, is_segm_task):
    for name, param in model.named_parameters():
        if is_segm_task and 'segmentation' not in name or not is_segm_task and 'depth' not in name:
            param.requires_grad = False


def log_config_to_clearml(task: Task, config: Config) -> None:
    """
    Log configuration parameters to ClearML.
    
    Args:
        task: ClearML Task object
        config: Configuration object containing all parameters
    """

    # Log model configuration
    task.connect({
        "Model": {
            "name": config.model.name,
            "encoder": config.model.encoder,
            "weights": config.model.weights
        },
        "Task": {
            "type": config.task.type,
            "mode": config.task.mode,
            "classes": config.task.classes
        },
        "Data": {
            "train_path": config.data.train_path,
            "valid_path": config.data.valid_path,
            "image_size": {
                "height": config.data.image_size.height,
                "width": config.data.image_size.width
            }
        },
        "Training": {
            "batch_size": config.training.batch_size,
            "epochs": config.training.epochs,
            "optimizer": {
                "name": config.training.optimizer.name,
                "lr": config.training.optimizer.lr
            },
            "scheduler": {
                "name": config.training.scheduler.name
            },
            "loss": {
                "name": config.training.loss.name,
                "weights": config.training.loss.weights,
                "ignore_index": config.training.loss.ignore_index
            },
            "early_stopping": config.training.early_stopping,
            "seed": config.training.seed
        },
        "Hardware": {
            "device": config.hardware.device,
            "parallel": {
                "type": config.hardware.parallel.type,
                "devices": config.hardware.parallel.device_ids
            }
        }
    })