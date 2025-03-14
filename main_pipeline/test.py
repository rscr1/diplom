import argparse
import os

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection, JaccardIndex, Dice
from tqdm import tqdm
from utils_func import visualize, classes2rgb, find_path

from config import Config
import dataset
from metrics.mse import MeanAbsoluteError, MeanSquaredError
from multihead_model import *
from utils_func import get_mask, get_depth


INTERPOLATION = {
    'INTER_NEAREST': cv2.INTER_NEAREST,
    'INTER_LINEAR': cv2.INTER_LINEAR,
    'INTER_CUBIC': cv2.INTER_CUBIC,
    'INTER_AREA': cv2.INTER_AREA,
    'INTER_LANCZOS4': cv2.INTER_LANCZOS4
}


def main(config: Config):
    # Setup paths
    path, _ = find_path(config.output.save_dir)
    print(f"Results will be saved to: {path}")
    
    # Создаем отдельные директории для разных типов предсказаний
    if config.task.type in ['depth', 'both']:
        depth_preds_path = os.path.join(path, 'depth_predictions')
        os.makedirs(depth_preds_path, exist_ok=True)
    
    if config.task.type in ['segm', 'both']:
        segm_preds_path = os.path.join(path, 'segm_predictions')
        os.makedirs(segm_preds_path, exist_ok=True)

    # Load colors for segmentation visualization if needed
    colors = None
    if config.task.type in ['segm', 'both']:
        colors = np.load(config.data.colors_path)

    # Create datasets and dataloaders
    valid_transforms = A.Compose([
        A.Resize(
            height=config.data.image_size.height, 
            width=config.data.image_size.width, 
            p=1.0, 
            interpolation=INTERPOLATION[config.data.augmentation.valid.resize.params.interpolation]),
        A.Normalize(p=1.0),
        ToTensorV2(p=1.0)
        ])

    # Initialize appropriate dataset based on task
    if config.task.type == 'segm':
        test_dataset = dataset.CustomSegmDataset(config.data.valid_path)  # Для оригинальных изображений
        valid_dataset = dataset.CustomSegmDataset(config.data.valid_path, valid_transforms)  # Для преобразованных изображений
    elif config.task.type == 'depth':
        test_dataset = dataset.CustomDepthDataset(config.data.valid_path)  # Для оригинальных изображений
        valid_dataset = dataset.CustomDepthDataset(config.data.valid_path, valid_transforms)  # Для преобразованных изображений
    else:
        test_dataset = dataset.CustomMultiDataset(config.data.valid_path)  # Для оригинальных изображений
        valid_dataset = dataset.CustomMultiDataset(config.data.valid_path, valid_transforms)  # Для преобразованных изображений

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config.testing.batch_size,
        shuffle=False,
        drop_last=False
    )

    if config.model.name == 'segm_fpn_depth_head':
        model = SegmFPNWithDepthHead(
            encoder_name=config.model.encoder,
            classes=config.task.classes
        )
    elif config.model.name == 'segm_fpn_depth_decoder_head':
        model = SegmFPNWithDepthDecoderHead(
            encoder_name=config.model.encoder,
            classes=config.task.classes
        )
    elif config.model.name == 'segm_deeplab_depth_head':
        model = SegmDeepLabV3WithDepthHead(
            encoder_name=config.model.encoder,
            classes=config.task.classes
        )
    elif config.model.name == 'segm_deeplab_depth_decoder_head':
        model = SegmDeepLabV3WithDepthDecoderHead(
            encoder_name=config.model.encoder,
            classes=config.task.classes
        )
    elif config.model.name == 'depth_fpn_segm_head':
        model = DepthFPNWithSegmHead(
            encoder_name=config.model.encoder,
            classes=config.task.classes
        )
    elif config.model.name == 'depth_fpn_segm_decoder_head':
        model = DepthFPNWithSegmDecoderHead(
            encoder_name=config.model.encoder,
            classes=config.task.classes
        )
    elif config.model.name == 'depth_deeplab_segm_head':
        model = DepthDeepLabV3WithSegmHead(
            encoder_name=config.model.encoder,
            classes=config.task.classes
        )
    elif config.model.name == 'depth_deeplab_segm_decoder_head':
        model = DepthDeepLabV3WithSegmDecoderHead(
            encoder_name=config.model.encoder,
            classes=config.task.classes
        )
    elif config.model.name == 'deeplabv3':
        model = smp.DeepLabV3(
            encoder_name=config.model.encoder,
            classes=config.task.classes,
            upsampling=32
        )
    elif config.model.name == 'fpn':
        model = smp.FPN(
            encoder_name=config.model.encoder,
            classes=config.task.classes
        )
    elif config.model.name == 'deeplabv3+':
        model = smp.DeepLabV3Plus(
            encoder_name=config.model.encoder,
            classes=config.task.classes,
            upsampling=32
        )

    model = model.to(config.testing.device)
    model.load_state_dict(torch.load(config.model.weights))
    
    if config.task.type in ['depth', 'both']:
        depth_metrics = MetricCollection([
            MeanSquaredError(squared=False).to(config.testing.device),
            MeanAbsoluteError().to(config.testing.device)
        ])
    
    if config.task.type in ['segm', 'both']:
        segm_metrics = MetricCollection([
            JaccardIndex(num_classes=config.task.classes, average='none', ignore_index=config.testing.ignore_index).to(config.testing.device),
            Dice(num_classes=config.task.classes, average='none', ignore_index=config.testing.ignore_index).to(config.testing.device)
        ])

    # Combined testing and visualization
    model.eval()
    
    if config.task.type in ['depth', 'both']:
        total_depth_metrics = torch.zeros((2, 1), device=config.testing.device)
        depth_metrics_names = ['MeanSquaredError', 'MeanAbsoluteError']
    
    if config.task.type in ['segm', 'both']:
        total_segm_metrics = torch.zeros((2, config.task.classes - 1), device=config.testing.device)
        segm_metrics_names = ['JaccardIndex', 'Dice']
    
    print("Running evaluation and generating visualizations...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(valid_dataloader)):
            inputs, labels = batch
            inputs, labels = inputs.to(config.testing.device), labels.to(config.testing.device)

            # Get model predictions
            out = model(inputs)
            if config.task.type == 'segm':
                mask = get_mask(out)
            elif config.task.type == 'depth':
                depth = get_depth(out)
                depth = depth[:, 0, :, :]
            else:
                depth, mask = out

            if config.task.type in ['depth', 'both']:
                depth_metrics_dict = depth_metrics(depth, labels)
                for i, name in enumerate(depth_metrics_names):
                    total_depth_metrics[i] += depth_metrics_dict[name]

            if config.task.type in ['segm', 'both']:
                mask = mask.argmax(1)
                segm_metrics_dict = segm_metrics(mask, labels)
                total_segm_metrics[0] += torch.nan_to_num(segm_metrics_dict[segm_metrics_names[0]], 0)
                total_segm_metrics[1] += torch.nan_to_num(segm_metrics_dict[segm_metrics_names[1]][1:], 0)

            # Generate visualizations for all samples in batch
            for i in range(len(inputs)):
                global_idx = batch_idx * config.testing.batch_size + i

                # Get original image and ground truth
                orig_image, orig_gt = test_dataset[global_idx]
                height, width = orig_image.shape[:2]

                # Определяем постпроцессинг для возврата к оригинальному размеру
                postprocessing = A.Resize(
                    height=height, 
                    width=width, 
                    p=1.0, 
                    interpolation=INTERPOLATION[config.data.augmentation.valid.resize.params.interpolation]
                )

                if config.task.type == 'depth':
                    # Визуализация только глубины
                    pred_depth = depth[i].cpu().numpy()
                    pred_depth = postprocessing(image=pred_depth)['image']
                    results_path = os.path.join(depth_preds_path, f'depth_sample_{global_idx:04d}.png')
                    visualize(
                        results_path,
                        task='depth',
                        image=orig_image,
                        ground_truth=orig_gt,
                        prediction=pred_depth
                    )
                elif config.task.type == 'segm':
                    # Визуализация только сегментации
                    pred_mask = mask[i].cpu().numpy()
                    pred_mask = postprocessing(image=pred_mask)['image']
                    pred_rgb = classes2rgb(pred_mask, colors)
                    gt_rgb = classes2rgb(orig_gt, colors)
                    results_path = os.path.join(segm_preds_path, f'segm_sample_{global_idx:04d}.png')
                    visualize(
                        results_path,
                        task='segm',
                        image=orig_image,
                        ground_truth_mask=gt_rgb,
                        predicted_mask=pred_rgb
                    )
                else:  # both
                    # Визуализация обоих предсказаний в одном изображении
                    pred_depth = depth[i].cpu().numpy()
                    pred_depth = postprocessing(image=pred_depth)['image']
                    
                    pred_mask = mask[i].cpu().numpy()
                    pred_mask = postprocessing(image=pred_mask)['image']
                    pred_rgb = classes2rgb(pred_mask, colors)
                    gt_rgb = classes2rgb(orig_gt, colors)
                    
                    # Сохраняем комбинированную визуализацию
                    results_path = os.path.join(path, f'multitask_sample_{global_idx:04d}.png')
                    visualize(
                        results_path,
                        task='both',
                        image=orig_image,
                        depth_gt=orig_gt,
                        depth_pred=pred_depth,
                        segm_gt=gt_rgb,
                        segm_pred=pred_rgb
                    )

    # Print results based on task
    print(f"\nTest Results:")
    
    if config.task.type in ['depth', 'both']:
        total_depth_metrics = total_depth_metrics / len(valid_dataloader)
        print(f"\nDepth Metrics:")
        print(f"RMSE: {total_depth_metrics[0].item():.4f}")
        print(f"MAE: {total_depth_metrics[1].item():.4f}")
    
    if config.task.type in ['segm', 'both']:
        total_segm_metrics = total_segm_metrics / len(valid_dataloader)
        print(f"\nSegmentation Metrics:")
        print(f"IoU per class: {total_segm_metrics[0].cpu().numpy()}")
        print(f"Mean IoU: {total_segm_metrics[0].mean().item():.4f}")
        print(f"Dice per class: {total_segm_metrics[1].cpu().numpy()}")
        print(f"Mean Dice: {total_segm_metrics[1].mean().item():.4f}")
    
    print(f"\nResults saved to {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Testing pipeline for depth estimation and segmentation")
    parser.add_argument("--config", type=str, help="Path to YAML config file", default="/AkhmetzyanovD/projects/nztfm/configs/segm_test_config.yaml")
    args = parser.parse_args()

    # Load and validate configuration
    config = Config.test_from_yaml(args.config)
    config.test_validate()
    
    
    main(
        config=config
    )