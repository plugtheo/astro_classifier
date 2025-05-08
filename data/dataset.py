"""
This module implements the data loading and preprocessing pipeline for astronomical images.
It includes dataset management, data augmentation, and efficient data loading.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import json
import os
from config.config import astro_config
import cv2

def resize_if_needed(img, **kwargs):
    """Resize image to 1280x1280 if not already that size."""
    try:
        # Convert PIL Image to numpy array if needed
        if isinstance(img, Image.Image):
            img = np.array(img)
        
        # Check if img is valid
        if not isinstance(img, np.ndarray):
            raise ValueError(f"Expected numpy array or PIL Image, got {type(img)}")
        
        # Check if image has valid shape
        if len(img.shape) < 2:
            raise ValueError(f"Invalid image shape: {img.shape}")
            
        if img.shape[:2] != (1280, 1280):
            # Use cv2.resize instead of A.resize
            return cv2.resize(img, (1280, 1280), interpolation=cv2.INTER_AREA)
        return img
    except Exception as e:
        print(f"Error in resize_if_needed: {str(e)}, Image type: {type(img)}, Shape: {getattr(img, 'shape', 'No shape')}, dtype: {getattr(img, 'dtype', 'No dtype')}")
        raise

class AstroDataset(Dataset):
    """
    A PyTorch Dataset class for loading and preprocessing astronomical images.
    
    This class implements a dataset loader that handles astronomical images from the
    dataset using processed metadata.
    
    Key features:
    - Metadata-based loading
    - Data augmentation for training
    - Proper image normalization
    - Efficient data loading
    """
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[A.Compose] = None,
        max_samples: Optional[int] = None
    ):
        """
        Args:
            data_dir (str): Directory containing the dataset
            split (str): Dataset split ('train', 'val', or 'test')
            transform (Optional[A.Compose]): Albumentations transforms
            max_samples (Optional[int]): Maximum number of samples to load
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform or self._get_default_transforms()
        
        # Load metadata
        self.metadata = self._load_metadata(split)
        
        # Filter out empty files
        self.metadata = [
            sample for sample in self.metadata 
            if (self.data_dir / sample['image_path']).stat().st_size > 0
        ]
        
        if max_samples:
            self.metadata = self.metadata[:max_samples]
            
        # Load class mapping
        self.class_mapping = self._load_class_mapping()
            
    def _load_metadata(self, split: str) -> Dict:
        """Load metadata for the specified split."""
        metadata_path = Path(astro_config.config.data_dir) / 'metadata' / f'{split}_metadata.json'
        with open(metadata_path, 'r') as f:
            return json.load(f)
            
    def _load_class_mapping(self) -> Dict[str, int]:
        """
        Load class name to index mapping.
        
        Returns:
            Dict[str, int]: Class mapping
        """
        mapping_path = Path(astro_config.config.data_dir) / 'metadata' / 'class_mapping.json'
        with open(mapping_path, 'r') as f:
            return json.load(f)
            
    def _get_default_transforms(self) -> A.Compose:
        """
        Get default transforms for the dataset.
        
        Returns:
            A.Compose: Albumentations transforms
        """
        return A.Compose([
            A.Lambda(image=resize_if_needed),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
            
    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        return len(self.metadata)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Sample index
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Image and label
        """
        sample = self.metadata[idx]
        
        # Load image
        image_path = self.data_dir / sample['image_path']
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Convert to numpy array first
                img_np = np.array(img, dtype=np.uint8)  # Changed to uint8
                
                # Apply transforms
                if self.transform:
                    try:
                        transformed = self.transform(image=img_np)
                        img = transformed['image']
                    except Exception as e:
                        print(f"Transform error for {image_path}: {str(e)}")
                        print(f"Image shape: {img_np.shape}, dtype: {img_np.dtype}")
                        raise
                
                # Clear numpy array to free memory
                del img_np
                
                # Get label
                label = self.class_mapping[sample['class']]
                
                return img, torch.tensor(label, dtype=torch.long)
        
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            # Return a zero tensor as fallback
            return torch.zeros((3, 1024, 1024)), torch.tensor(0, dtype=torch.long)

def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
    persistent_workers: bool = True
) -> Dict[str, DataLoader]:
    """Create data loaders for training, validation, and testing."""
    # Create datasets
    train_dataset = AstroDataset(
        data_dir=data_dir,
        split='train',
        max_samples=max_samples
    )
    val_dataset = AstroDataset(
        data_dir=data_dir,
        split='val',
        max_samples=max_samples
    )
    test_dataset = AstroDataset(
        data_dir=data_dir,
        split='test',
        max_samples=max_samples
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    } 