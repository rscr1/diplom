import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import AugmentationSetConfig


def _get_interpolation(interpolation_str: str) -> int:
    """Convert interpolation string to OpenCV constant."""
    interpolation_map = {
        'INTER_NEAREST': cv2.INTER_NEAREST,
        'INTER_LINEAR': cv2.INTER_LINEAR,
        'INTER_CUBIC': cv2.INTER_CUBIC,
        'INTER_AREA': cv2.INTER_AREA,
        'INTER_LANCZOS4': cv2.INTER_LANCZOS4
    }
    return interpolation_map.get(interpolation_str, cv2.INTER_NEAREST)


def _create_transforms(aug_config: AugmentationSetConfig, height: int, width: int) -> A.Compose:
    """Create albumentations transforms from config."""
    transforms = []
    
    # Add resize if enabled
    if aug_config.resize and aug_config.resize.enable:
        interpolation = _get_interpolation(aug_config.resize.params.interpolation)
        transforms.append(
            A.Resize(height=height, width=width, interpolation=interpolation, p=1.0)
        )
    
    # Add horizontal flip if enabled
    if aug_config.horizontal_flip and aug_config.horizontal_flip.enable:
        p = aug_config.horizontal_flip.params.p if aug_config.horizontal_flip.params else 0.5
        transforms.append(A.HorizontalFlip(p=p))
    
    # Add shift scale rotate if enabled
    if aug_config.shift_scale_rotate and aug_config.shift_scale_rotate.enable:
        params = aug_config.shift_scale_rotate.params
        transforms.append(
            A.ShiftScaleRotate(
                rotate_limit=params.rotate_limit,
                border_mode=params.border_mode,
                value=params.value,
                p=params.p
            )
        )
    
    # Add brightness contrast if enabled
    if aug_config.brightness_contrast and aug_config.brightness_contrast.enable:
        params = aug_config.brightness_contrast.params
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=params.brightness_limit,
                contrast_limit=params.contrast_limit,
                p=params.p
            )
        )
    
    # Add grid distortion if enabled
    if aug_config.grid_distortion and aug_config.grid_distortion.enable:
        params = aug_config.grid_distortion.params
        transforms.append(
            A.GridDistortion(
                num_steps=params.num_steps,
                distort_limit=params.distort_limit,
                interpolation=cv2.INTER_NEAREST,
                p=params.p
            )
        )
    
    # Add normalize if enabled
    if aug_config.normalize and aug_config.normalize.enable:
        transforms.append(A.Normalize(p=1.0))
    
    # Add to tensor if enabled
    if aug_config.to_tensor and aug_config.to_tensor.enable:
        transforms.append(ToTensorV2(p=1.0))
    
    return A.Compose(transforms)


def augment(config, height: int, width: int):
    """
    Create train and validation transforms based on configuration.
    
    Args:
        config: Augmentation configuration
        height: Target height
        width: Target width
    
    Returns:
        tuple: (train_transforms, valid_transforms)
    """
    train_transforms = _create_transforms(config.train, height, width)
    valid_transforms = _create_transforms(config.valid, height, width)
    
    return train_transforms, valid_transforms
