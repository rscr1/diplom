"""
Configuration module for handling YAML config files.
"""
from dataclasses import dataclass
import os
from typing import Dict, Optional, List
import yaml


@dataclass
class ModelConfig:
    name: str
    encoder: str
    weights: Optional[str]


@dataclass
class TaskConfig:
    type: str
    mode: str
    classes: int


@dataclass
class ImageSizeConfig:
    height: int
    width: int


@dataclass
class AugmentationParamsConfig:
    interpolation: Optional[str] = None
    p: Optional[float] = None
    rotate_limit: Optional[List[int]] = None
    border_mode: Optional[int] = None
    value: Optional[int] = None
    brightness_limit: Optional[List[float]] = None
    contrast_limit: Optional[List[float]] = None
    num_steps: Optional[int] = None
    distort_limit: Optional[List[float]] = None


@dataclass
class AugmentationConfig:
    enable: bool
    params: Optional[AugmentationParamsConfig] = None


@dataclass
class AugmentationSetConfig:
    resize: AugmentationConfig
    normalize: AugmentationConfig
    to_tensor: AugmentationConfig
    horizontal_flip: Optional[AugmentationConfig] = None
    shift_scale_rotate: Optional[AugmentationConfig] = None
    brightness_contrast: Optional[AugmentationConfig] = None
    grid_distortion: Optional[AugmentationConfig] = None


@dataclass
class AugmentationsConfig:
    train: AugmentationSetConfig
    valid: AugmentationSetConfig


@dataclass
class DataConfig:
    train_path: str
    valid_path: str
    colors_path: Optional[str]
    image_size: ImageSizeConfig
    augmentation: AugmentationsConfig


@dataclass
class OptimizerConfig:
    name: str
    lr: float


@dataclass
class SchedulerConfig:
    name: Optional[str]
    mode: Optional[str]
    factor: Optional[float]
    patience: Optional[int]
    step_size: Optional[int]
    gamma: Optional[float]
    max_lr: Optional[float]


@dataclass
class LossConfig:
    name: str
    weights: Optional[list]
    ignore_index: Optional[int]


@dataclass
class MetricsConfig:
    segmentation: List[str]
    depth: List[str]


@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    loss: LossConfig
    metrics: MetricsConfig
    early_stopping: bool
    seed: int


@dataclass
class TestingConfig:
    batch_size: int
    device: str
    metrics: MetricsConfig
    ignore_index: Optional[int]



@dataclass
class ParallelConfig:
    type: str
    device_ids: Optional[list]


@dataclass
class HardwareConfig:
    device: str
    parallel: ParallelConfig


@dataclass
class OutputConfig:
    project_name: str
    save_dir: str


@dataclass
class Config:
    model: ModelConfig
    task: TaskConfig
    data: DataConfig
    hardware: HardwareConfig
    output: OutputConfig
    training: Optional[TrainingConfig] = None
    testing: Optional[TestingConfig] = None

    @classmethod
    def train_from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from YAML file."""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Validate and create nested dataclass objects
        model_config = ModelConfig(**config_dict['model'])
        task_config = TaskConfig(**config_dict['task'])
        
        image_size = ImageSizeConfig(**config_dict['data']['image_size'])
        
        # Process augmentation configurations
        def create_aug_config(aug_dict: Dict) -> AugmentationConfig:
            if aug_dict is None:
                return None
            params = AugmentationParamsConfig(**aug_dict.get('params', {})) if 'params' in aug_dict else None
            return AugmentationConfig(enable=aug_dict['enable'], params=params)

        def create_aug_set_config(aug_set_dict: Dict) -> AugmentationSetConfig:
            return AugmentationSetConfig(
                resize=create_aug_config(aug_set_dict.get('resize')),
                normalize=create_aug_config(aug_set_dict.get('normalize')),
                to_tensor=create_aug_config(aug_set_dict.get('to_tensor')),
                horizontal_flip=create_aug_config(aug_set_dict.get('horizontal_flip')),
                shift_scale_rotate=create_aug_config(aug_set_dict.get('shift_scale_rotate')),
                brightness_contrast=create_aug_config(aug_set_dict.get('brightness_contrast')),
                grid_distortion=create_aug_config(aug_set_dict.get('grid_distortion'))
            )

        augmentations = AugmentationsConfig(
            train=create_aug_set_config(config_dict['data']['augmentation']['train']),
            valid=create_aug_set_config(config_dict['data']['augmentation']['valid'])
        )
        
        data_config = DataConfig(
            train_path=config_dict['data']['train_path'],
            valid_path=config_dict['data']['valid_path'],
            colors_path=config_dict['data']['colors_path'],
            image_size=image_size,
            augmentation=augmentations
        )
        
        optimizer_config = OptimizerConfig(**config_dict['training']['optimizer'])
        scheduler_config = SchedulerConfig(**config_dict['training']['scheduler'])
        loss_config = LossConfig(**config_dict['training']['loss'])
        metrics_config = MetricsConfig(**config_dict['training']['metrics'])
        training_config = TrainingConfig(**{
            **config_dict['training'],
            'optimizer': optimizer_config,
            'scheduler': scheduler_config,
            'loss': loss_config,
            'metrics': metrics_config
        })
        
        parallel_config = ParallelConfig(**config_dict['hardware']['parallel'])
        hardware_config = HardwareConfig(**{**config_dict['hardware'], 'parallel': parallel_config})
        
        output_config = OutputConfig(**config_dict['output'])

        return cls(
            model=model_config,
            task=task_config,
            data=data_config,
            training=training_config,
            hardware=hardware_config,
            output=output_config
        )
    

    @classmethod
    def test_from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from YAML file for testing."""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Validate and create nested dataclass objects
        model_config = ModelConfig(**config_dict['model'])
        task_config = TaskConfig(**config_dict['task'])
        
        image_size = ImageSizeConfig(**config_dict['data']['image_size'])
        
        # Process augmentation configurations
        def create_aug_config(aug_dict: Dict) -> AugmentationConfig:
            if aug_dict is None:
                return None
            params = AugmentationParamsConfig(**aug_dict.get('params', {})) if 'params' in aug_dict else None
            return AugmentationConfig(enable=aug_dict['enable'], params=params)

        def create_aug_set_config(aug_set_dict: Dict) -> AugmentationSetConfig:
            return AugmentationSetConfig(
                resize=create_aug_config(aug_set_dict.get('resize')),
                normalize=create_aug_config(aug_set_dict.get('normalize')),
                to_tensor=create_aug_config(aug_set_dict.get('to_tensor')),
                horizontal_flip=None,
                shift_scale_rotate=None,
                brightness_contrast=None,
                grid_distortion=None
            )

        augmentations = AugmentationsConfig(
            train=None,
            valid=create_aug_set_config(config_dict['data']['augmentation'])
        )
        
        data_config = DataConfig(
            train_path=None,
            valid_path=config_dict['data']['test_path'],
            colors_path=config_dict['data']['colors_path'],
            image_size=image_size,
            augmentation=augmentations
        )
        
        metrics_config = MetricsConfig(**config_dict['testing']['metrics'])
        testing_config = TestingConfig(
            batch_size=config_dict['testing']['batch_size'],
            device=config_dict['testing']['device'],
            metrics=metrics_config,
            ignore_index=config_dict['testing'].get('ignore_index')
        )
        
        hardware_config = HardwareConfig(
            device=config_dict['testing']['device'],
            parallel=ParallelConfig(type=None, device_ids=None)
        )
        
        output_config = OutputConfig(
            save_dir=config_dict['output']['save_dir'],
            project_name=None
        )

        return cls(
            model=model_config,
            task=task_config,
            data=data_config,
            testing=testing_config,
            hardware=hardware_config,
            output=output_config
        )
    

    def train_validate(self) -> None:
        """Validate configuration values."""
        # Validate image size
        assert self.data.image_size.height > 0, "Image height must be positive"
        assert self.data.image_size.width > 0, "Image width must be positive"
        
        # Validate paths
        assert os.path.exists(self.data.train_path), f"Train path not found: {self.data.train_path}"
        assert os.path.exists(self.data.valid_path), f"Valid path not found: {self.data.valid_path}"
        if self.data.colors_path:
            assert os.path.exists(self.data.colors_path), f"Colors path not found: {self.data.colors_path}"
            
        # Validate training parameters
        assert self.training.batch_size > 0, "Batch size must be positive"
        assert self.training.epochs > 0, "Number of epochs must be positive"
        assert self.task.classes > 0, "Number of classes must be positive"
        
        # Validate task type
        assert self.task.type in {'depth', 'segm'}, "Task type must be 'depth' or 'segm'"
        if self.task.type == 'segm':
            assert self.task.mode in {'binary', 'multiclass', 'multilabel'}, "Invalid segmentation mode"
            assert all(m in {'iou', 'dice'} for m in self.training.metrics.segmentation), "Invalid segmentation metrics"
        else:
            assert all(m in {'mse', 'mae', 'rmse'} for m in self.training.metrics.depth), "Invalid depth metrics"

        # Validate model name
        valid_models = {
            'deeplabv3', 'fpn', 'deeplabv3+',
            'segm_fpn_depth_head', 'segm_fpn_depth_decoder_head',
            'segm_deeplab_depth_head', 'segm_deeplab_depth_decoder_head',
            'depth_fpn_segm_head', 'depth_fpn_segm_decoder_head',
            'depth_deeplab_segm_head', 'depth_deeplab_segm_decoder_head'
        }
        assert self.model.name in valid_models, f"Unknown model name: {self.model.name}"

        # Validate Hydranet models weights requirement
        hydranet_models = {
            'segm_fpn_depth_head', 'segm_fpn_depth_decoder_head',
            'segm_deeplab_depth_head', 'segm_deeplab_depth_decoder_head',
            'depth_fpn_segm_head', 'depth_fpn_segm_decoder_head',
            'depth_deeplab_segm_head', 'depth_deeplab_segm_decoder_head'
        }
        if self.model.name in hydranet_models:
            assert self.model.weights is not None, f"{self.model.name} model needs previous model weights"

        # Validate hardware configuration
        valid_parallel_types = {'ddp', 'dp', None}
        assert self.hardware.parallel.type in valid_parallel_types, f"Unknown hardware parallel type: {self.hardware.parallel.type}"

        # Validate optimizer
        valid_optimizers = {'Adam', 'AdamW', 'SGD'}
        assert self.training.optimizer.name in valid_optimizers, f"Unknown optimizer name: {self.training.optimizer.name}"

        # Validate scheduler
        if self.training.scheduler.name is not None:
            valid_schedulers = {'StepLR', 'OneCycleLR', 'ReduceLROnPlateau'}
            assert self.training.scheduler.name in valid_schedulers, f"Unknown scheduler name: {self.training.scheduler.name}"

        # Validate loss function
        if self.task.type == 'segm':
            valid_losses = {'focal', 'crossentropy', 'lovasz', 'dice', 'jaccard', 'tversky'}
            if isinstance(self.training.loss.name, str):
                assert self.training.loss.name in valid_losses, f"Unknown loss name for segmentation: {self.training.loss.name}"
            elif isinstance(self.training.loss.name, list) or isinstance(self.training.loss.name, tuple):
                assert all(loss in valid_losses for loss in self.training.loss.name)
        else:  # depth task
            assert self.training.loss.name == 'mse', f"Unknown loss name for depth estimation: {self.training.loss.name}" 


    def test_validate(self) -> None:
        """Validate configuration values."""
        # Validate image size
        assert self.data.image_size.height > 0, "Image height must be positive"
        assert self.data.image_size.width > 0, "Image width must be positive"
        
        # Validate paths
        assert os.path.exists(self.data.valid_path), f"Valid path not found: {self.data.valid_path}"
        if self.task.type == 'segm':
            assert os.path.exists(self.data.colors_path), f"Colors path not found: {self.data.colors_path}"
            
        # Validate testing parameters
        assert self.testing.batch_size > 0, "Batch size must be positive"
        assert self.task.classes > 0, "Number of classes must be positive"
        
        # Validate task type
        assert self.task.type in {'depth', 'segm', 'both'}, "Task type must be 'depth' or 'segm'"
        if self.task.type in {'segm', 'both'}:
            assert self.task.mode in {'binary', 'multiclass', 'multilabel'}, "Invalid segmentation mode"
            assert all(m in {'iou', 'dice'} for m in self.testing.metrics.segmentation), "Invalid segmentation metrics"
        if self.task.type in {'depth', 'both'}:
            assert all(m in {'mse', 'mae', 'rmse'} for m in self.testing.metrics.depth), "Invalid depth metrics"

        # Validate model name
        valid_models = {
            'deeplabv3', 'fpn', 'deeplabv3+',
            'segm_fpn_depth_head', 'segm_fpn_depth_decoder_head',
            'segm_deeplab_depth_head', 'segm_deeplab_depth_decoder_head',
            'depth_fpn_segm_head', 'depth_fpn_segm_decoder_head',
            'depth_deeplab_segm_head', 'depth_deeplab_segm_decoder_head'
        }
        assert self.model.name in valid_models, f"Unknown model name: {self.model.name}"

        # Validate models weights requirement
        assert self.model.weights is not None, f"{self.model.name} model needs previous model weights"