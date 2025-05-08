"""
This module implements the configuration management system for the astronomical classifier.
It includes model, data, and training configurations with well-documented hyperparameters.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import os
from enum import Enum
import torch

class TrainingTask(Enum):
    """Available training tasks."""
    ASTRO_CLASSIFIER = "astro_classifier"

@dataclass
class BaseConfig:
    """Base configuration shared across all training tasks."""
    # Data loading configurations
    num_workers: int = 4  # Reduced workers to save memory
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True
    
    # Image preprocessing configurations
    image_size: int = 1280  # Updated to match dataset resizing
    normalize_mean: tuple = (0.485, 0.456, 0.406)
    normalize_std: tuple = (0.229, 0.224, 0.225)
    
    # Training process configurations
    precision: str = '16-mixed'  # Changed to string format for PyTorch Lightning
    gradient_clip_val: float = 1.0
    max_epochs: int = 100
    early_stopping_patience: int = 10
    
    # GPU optimization configurations
    accelerator: str = 'gpu'  # Hardware accelerator to use. 'gpu' for NVIDIA GPUs, 'cpu' for CPU training.
    devices: int = 1  # Number of GPUs to use. More GPUs enable distributed training but require more memory.
    strategy: str = 'auto'  # Distributed training strategy. 'auto' lets PyTorch Lightning choose the best approach.
    sync_batchnorm: bool = True  # Synchronizes batch normalization across GPUs. Important for multi-GPU training.
    benchmark: bool = True  # Enables cuDNN benchmarking. Can speed up training but uses more memory.
    deterministic: bool = False  # Makes training deterministic. Slows down training but ensures reproducibility.
    
    # Logging and checkpointing configurations
    log_dir: str = 'logs'  # Directory for storing training logs. Used for monitoring training progress.
    checkpoint_dir: str = 'checkpoints'  # Directory for saving model checkpoints. Important for model recovery.
    save_top_k: int = 3  # Number of best models to keep. Saves disk space while maintaining best models.
    save_last: bool = True  # Whether to save the last model. Useful for resuming training.
    log_every_n_steps: int = 50  # Frequency of logging. Balances between monitoring and performance.
    val_check_interval: float = 0.25

@dataclass
class AstroClassifierConfig(BaseConfig):
    """Configuration specific to the main astronomical classifier."""
    # Model architecture configurations
    backbone: str = 'resnet18'  # Changed to ResNet18
    pretrained: bool = True
    
    # Classification head configurations
    num_classes: int = 8
    class_names: List[str] = field(default_factory=lambda: [
        'asteroid',
        'black_hole',
        'comet',
        'constellation',
        'galaxy',
        'nebula',
        'planet',
        'star'
    ])
    
    # Class weights for handling imbalance
    class_weights: Dict[str, float] = field(default_factory=lambda: {
        'asteroid': 1.4,      # Increased from 1.2 due to 74.34% accuracy
        'black_hole': 1.5,    # Reduced from 2.0 as it's performing well (90%)
        'comet': 2.0,         # Increased from 1.5 due to 67.15% accuracy
        'constellation': 2.5, # Increased from 1.0 due to 50.96% accuracy
        'galaxy': 1.0,        # Kept at 1.0 as it's performing well
        'nebula': 1.0,        # Reduced from 1.3 as it's performing well (84.22%)
        'planet': 0.8,        # Reduced from 1.0 as it's performing well (93%)
        'star': 0.8          # Kept at 0.8 as it's performing well
    })
    
    # Training parameter configurations
    batch_size: int = 24  # Reduced from 32 to be more memory efficient
    learning_rate: float = 1e-4  # Base learning rate (will be modified by OneCycleLR)
    weight_decay: float = 1e-4
    gradient_accumulation_steps: int = 3  # Increased from 2 for effective batch size of 72
    num_workers: int = 4  # Kept at 4 for good data loading performance
    precision: str = '16-mixed'  # Keep mixed precision for efficiency
    gradient_clip_val: float = 0.5  # Reduced from 1.0 for better stability
    early_stopping_patience: int = 15  # Increased from 10 for more stable training
    max_epochs: int = 100
    val_check_interval: float = 0.25
    save_top_k: int = 3
    
    # Astronomical-specific augmentation configurations
    augmentation_config: Dict[str, Dict] = field(default_factory=lambda: {
        'asteroid': {
            'rotation_range': 15,      # Small rotation as asteroids maintain orientation
            'brightness_range': 0.2,   # Increased for better robustness
            'contrast_range': 0.2,     # Increased for better robustness
            'noise_level': 0.1,        # Increased for better robustness
            'flip': False              # No flips to maintain orientation
        },
        'black_hole': {
            'rotation_range': 0,       # No rotation to maintain physics
            'brightness_range': 0.2,   # Increased for better robustness
            'contrast_range': 0.2,     # Increased for better robustness
            'noise_level': 0.1,        # Increased for better robustness
            'flip': False              # No flips to maintain physics
        },
        'comet': {
            'rotation_range': 15,      # Increased for better robustness
            'brightness_range': 0.2,   # Increased for better robustness
            'contrast_range': 0.2,     # Increased for better robustness
            'noise_level': 0.1,        # Increased for better robustness
            'flip': False              # No flips to maintain tail direction
        },
        'constellation': {
            'rotation_range': 360,     # Full rotation as constellations are orientation-invariant
            'brightness_range': 0.2,   # Increased for better robustness
            'contrast_range': 0.2,     # Increased for better robustness
            'noise_level': 0.1,        # Increased for better robustness
            'flip': True               # Allow flips as constellations are pattern-based
        },
        'galaxy': {
            'rotation_range': 360,     # Full rotation as galaxies are orientation-invariant
            'brightness_range': 0.2,   # Increased for better robustness
            'contrast_range': 0.2,     # Increased for better robustness
            'noise_level': 0.1,        # Increased for better robustness
            'flip': True               # Allow flips as galaxies are symmetric
        },
        'nebula': {
            'rotation_range': 360,     # Full rotation as nebulae are orientation-invariant
            'brightness_range': 0.2,   # Increased for better robustness
            'contrast_range': 0.2,     # Increased for better robustness
            'noise_level': 0.1,        # Increased for better robustness
            'flip': True               # Allow flips as nebulae are often symmetric
        },
        'planet': {
            'rotation_range': 360,     # Full rotation as planets are spherical
            'brightness_range': 0.2,   # Increased for better robustness
            'contrast_range': 0.2,     # Increased for better robustness
            'noise_level': 0.1,        # Increased for better robustness
            'flip': True               # Allow flips as planets are symmetric
        },
        'star': {
            'rotation_range': 0,       # No rotation as stars are point sources
            'brightness_range': 0.2,   # Increased for better robustness
            'contrast_range': 0.2,     # Increased for better robustness
            'noise_level': 0.1,        # Increased for better robustness
            'flip': True               # Allow flips as stars are symmetric
        }
    })
    
    # Data path configurations
    data_dir: str = 'data/raw/SpaceNet.FLARE.imam_alam'  # Directory containing raw images
    raw_data_dir: str = 'data/raw'  # Directory containing raw images
    metadata_dir: str = 'data/astro_classifier/metadata'  # Directory containing metadata files

class ConfigManager:
    """Manages configuration for different training tasks."""
    def __init__(self, task: Union[str, TrainingTask] = TrainingTask.ASTRO_CLASSIFIER):
        if isinstance(task, str):
            task = TrainingTask(task)
            
        self.task = task
        self.config = self._get_task_config()
    
    def _get_task_config(self) -> AstroClassifierConfig:
        """Get configuration for the specified task."""
        configs = {
            TrainingTask.ASTRO_CLASSIFIER: AstroClassifierConfig()
        }
        return configs[self.task]
    
    def get_trainer_config(self) -> dict:
        """Get PyTorch Lightning trainer configuration."""
        return {
            'accelerator': self.config.accelerator,
            'devices': self.config.devices,
            'strategy': self.config.strategy,
            'precision': self.config.precision,
            'gradient_clip_val': self.config.gradient_clip_val,
            'accumulate_grad_batches': self.config.gradient_accumulation_steps,
            'max_epochs': self.config.max_epochs,
            'log_every_n_steps': self.config.log_every_n_steps,
            'val_check_interval': self.config.val_check_interval,
            'sync_batchnorm': self.config.sync_batchnorm,
            'benchmark': self.config.benchmark,
            'deterministic': self.config.deterministic
        }
    
    def get_dataloader_config(self) -> dict:
        """Get DataLoader configuration."""
        return {
            'batch_size': self.config.batch_size,
            'num_workers': self.config.num_workers,
            'pin_memory': self.config.pin_memory,
            'persistent_workers': self.config.persistent_workers,
            'prefetch_factor': self.config.prefetch_factor
        }

# Create default configuration instances
astro_config = ConfigManager(TrainingTask.ASTRO_CLASSIFIER) 