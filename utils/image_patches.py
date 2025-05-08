"""
Utility for handling large astronomical images and patches.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import torch
from torch.utils.data import Dataset
import logging
from tqdm import tqdm

class ImagePatcher:
    """
    A utility class for handling large astronomical images and patches.
    
    Key features:
    - Patch extraction from large images
    - Overlapping patches for better coverage
    - Patch reconstruction
    - Quality control for patches
    - Memory-efficient processing
    """
    
    def __init__(
        self,
        patch_size: Tuple[int, int] = (224, 224),
        stride: int = 112,  # 50% overlap
        min_patch_quality: float = 0.5,
        max_patches_per_image: int = 16
    ):
        """
        Initialize the image patcher.
        
        Args:
            patch_size (Tuple[int, int]): Size of patches to extract
            stride (int): Stride for patch extraction (controls overlap)
            min_patch_quality (float): Minimum quality threshold for patches
            max_patches_per_image (int): Maximum number of patches per image
        """
        self.patch_size = patch_size
        self.stride = stride
        self.min_patch_quality = min_patch_quality
        self.max_patches_per_image = max_patches_per_image
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def extract_patches(
        self,
        image: np.ndarray,
        metadata: Dict
    ) -> List[Tuple[np.ndarray, Dict]]:
        """
        Extract patches from a large image.
        
        Args:
            image (np.ndarray): Input image
            metadata (Dict): Image metadata
            
        Returns:
            List[Tuple[np.ndarray, Dict]]: List of (patch, patch_metadata) tuples
        """
        height, width = image.shape[:2]
        patches = []
        patch_metadata = []
        
        # Calculate patch positions
        for y in range(0, height - self.patch_size[0] + 1, self.stride):
            for x in range(0, width - self.patch_size[1] + 1, self.stride):
                # Extract patch
                patch = image[y:y + self.patch_size[0], x:x + self.patch_size[1]]
                
                # Check patch quality
                if self._check_patch_quality(patch):
                    # Create patch metadata
                    patch_meta = metadata.copy()
                    patch_meta.update({
                        'patch_id': f"{metadata['image_id']}_patch_{len(patches)}",
                        'patch_position': (x, y),
                        'patch_size': self.patch_size,
                        'original_size': (width, height)
                    })
                    
                    patches.append(patch)
                    patch_metadata.append(patch_meta)
                    
                    # Limit number of patches
                    if len(patches) >= self.max_patches_per_image:
                        break
                        
            if len(patches) >= self.max_patches_per_image:
                break
                
        return list(zip(patches, patch_metadata))
        
    def _check_patch_quality(self, patch: np.ndarray) -> bool:
        """
        Check if a patch meets quality criteria.
        
        Args:
            patch (np.ndarray): Image patch
            
        Returns:
            bool: True if patch meets quality criteria
        """
        # Check for blank or nearly blank patches
        if patch.mean() < 10 or patch.mean() > 245:
            return False
            
        # Check for sufficient detail
        if np.std(patch) < 20:
            return False
            
        # Check for sufficient non-zero pixels
        if (patch > 0).mean() < self.min_patch_quality:
            return False
            
        return True
        
    def reconstruct_image(
        self,
        patches: List[np.ndarray],
        patch_positions: List[Tuple[int, int]],
        original_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Reconstruct original image from patches.
        
        Args:
            patches (List[np.ndarray]): List of image patches
            patch_positions (List[Tuple[int, int]]): List of patch positions
            original_size (Tuple[int, int]): Original image size
            
        Returns:
            np.ndarray: Reconstructed image
        """
        height, width = original_size
        reconstructed = np.zeros((height, width, 3), dtype=np.float32)
        count = np.zeros((height, width), dtype=np.float32)
        
        for patch, (x, y) in zip(patches, patch_positions):
            h, w = patch.shape[:2]
            reconstructed[y:y+h, x:x+w] += patch
            count[y:y+h, x:x+w] += 1
            
        # Average overlapping regions
        count[count == 0] = 1  # Avoid division by zero
        reconstructed /= count[..., np.newaxis]
        
        return reconstructed.astype(np.uint8)

class PatchedAstroDataset(Dataset):
    """
    Dataset class for handling patched astronomical images.
    
    Key features:
    - Efficient patch loading
    - On-the-fly patch extraction
    - Memory-efficient processing
    - Quality control
    """
    
    def __init__(
        self,
        data_dir: Path,
        metadata: List[Dict],
        patch_size: Tuple[int, int] = (224, 224),
        transform: Optional[callable] = None,
        max_patches_per_image: int = 16
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir (Path): Directory containing images
            metadata (List[Dict]): List of image metadata
            patch_size (Tuple[int, int]): Size of patches
            transform (callable): Optional transform to apply to patches
            max_patches_per_image (int): Maximum patches per image
        """
        self.data_dir = data_dir
        self.metadata = metadata
        self.transform = transform
        self.patcher = ImagePatcher(
            patch_size=patch_size,
            max_patches_per_image=max_patches_per_image
        )
        
        # Pre-compute patch information
        self.patches = []
        self.patch_metadata = []
        
        for item in tqdm(metadata, desc="Preparing patches"):
            image_path = self.data_dir / f"{item['image_id']}.jpg"
            if not image_path.exists():
                continue
                
            try:
                image = np.array(Image.open(image_path))
                patches = self.patcher.extract_patches(image, item)
                
                self.patches.extend([p[0] for p in patches])
                self.patch_metadata.extend([p[1] for p in patches])
                
            except Exception as e:
                logging.error(f"Error processing {image_path}: {str(e)}")
                continue
                
    def __len__(self) -> int:
        return len(self.patches)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Get a patch and its metadata.
        
        Args:
            idx (int): Index of the patch
            
        Returns:
            Tuple[torch.Tensor, Dict]: (patch, metadata)
        """
        patch = self.patches[idx]
        metadata = self.patch_metadata[idx]
        
        if self.transform:
            patch = self.transform(patch)
            
        return patch, metadata 