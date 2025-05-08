"""
Utility functions for classification using the astronomical classifier model.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from models.astro_classifier import AstroClassifier
from utils.cam_utils import ClassActivationMap
from config.config import astro_config
import torchvision.ops as ops

from utils.model_utils import load_pretrained_model


class AstroClassifierPredictor:
    """
    A utility class for making predictions using the astronomical classifier.
    
    This class provides easy-to-use methods for classifying astronomical images
    using the trained model. It handles image preprocessing, model loading,
    and prediction post-processing.
    
    Key features:
    - Easy model loading and initialization
    - Advanced image preprocessing for astronomical images
    - Multi-task prediction
    - Uncertainty estimation
    - Class label mapping
    - Multi-scale processing for large images
    """
    def __init__(self, model_path=None):
        """Initialize the predictor with a trained model."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Initialize CAM generator
        self.cam_generator = ClassActivationMap(self.model)
        
        # Define image transformations for different sizes
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Store class names
        self.class_names = astro_config.config.class_names
    
    def _load_model(self, model_path):
        """Load the model from checkpoint."""
        model = AstroClassifier()
        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device)
            # Handle PyTorch Lightning checkpoint format
            if 'state_dict' in checkpoint:
                # Remove 'model.' prefix from state dict keys
                state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('model.')}
                model.load_state_dict(state_dict)
            else:
                # Try loading the checkpoint directly as state dict
                try:
                    model.load_state_dict(checkpoint)
                except Exception as e:
                    print(f"Error loading checkpoint: {e}")
                    print("Available keys in checkpoint:", checkpoint.keys())
                    raise
        model = model.to(self.device)
        return model
    
    def preprocess_image(self, image_path):
        """Preprocess image for prediction."""
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        
        # Store original size
        self.original_size = image.size
        print(f"\nOriginal image size: {self.original_size}")
        
        # Calculate the scale factor to maintain aspect ratio
        target_size = 224  # ResNet default input size
        min_size = min(image.size)
        max_size = max(image.size)
        
        # Adjust scale to ensure patches are at least 224x224
        if min_size < target_size:
            scale = target_size / min_size
        else:
            scale = 1.0
            
        new_w = int(image.size[0] * scale)
        new_h = int(image.size[1] * scale)
        print(f"Resized dimensions: {new_w}x{new_h}")
        
        # Resize image maintaining aspect ratio
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Convert to tensor and normalize
        img_tensor = self.transform(image)
        
        # Create patches if image is large
        if new_w > target_size or new_h > target_size:
            patches = []
            patch_info = []  # Store patch locations
            
            # Calculate optimal patch size and stride based on image dimensions
            # For very large images, use larger patches to reduce overlap
            if max_size > 2000:
                patch_size = min(448, max_size // 4)  # Larger patches for very large images
                stride = patch_size // 2  # 50% overlap
            else:
                patch_size = target_size
                stride = patch_size // 2  # 50% overlap
            
            # Ensure patch size is at least target_size
            patch_size = max(target_size, patch_size)
            
            # Calculate number of patches to create
            num_patches_w = max(1, (new_w - patch_size) // stride + 1)
            num_patches_h = max(1, (new_h - patch_size) // stride + 1)
            
            # Adjust stride if too many patches would be created
            max_patches = 20  # Maximum number of patches per dimension
            if num_patches_w * num_patches_h > max_patches * max_patches:
                # Calculate new stride to achieve desired number of patches
                stride_w = (new_w - patch_size) // (max_patches - 1)
                stride_h = (new_h - patch_size) // (max_patches - 1)
                stride = max(stride_w, stride_h)
            
            print(f"Creating patches with size {patch_size}x{patch_size} and stride {stride}")
            
            # Create patches with calculated stride
            for y in range(0, new_h - patch_size + 1, stride):
                for x in range(0, new_w - patch_size + 1, stride):
                    patch = img_tensor[:, y:y+patch_size, x:x+patch_size]
                    patches.append(patch)
                    patch_info.append({
                        'x': x,
                        'y': y,
                        'w': patch_size,
                        'h': patch_size,
                        'scale_x': self.original_size[0] / new_w,
                        'scale_y': self.original_size[1] / new_h
                    })
            
            # Add edge patches if needed
            if new_h % patch_size != 0:
                for x in range(0, new_w - patch_size + 1, stride):
                    patch = img_tensor[:, -patch_size:, x:x+patch_size]
                    patches.append(patch)
                    patch_info.append({
                        'x': x,
                        'y': new_h - patch_size,
                        'w': patch_size,
                        'h': patch_size,
                        'scale_x': self.original_size[0] / new_w,
                        'scale_y': self.original_size[1] / new_h
                    })
            
            if new_w % patch_size != 0:
                for y in range(0, new_h - patch_size + 1, stride):
                    patch = img_tensor[:, y:y+patch_size, -patch_size:]
                    patches.append(patch)
                    patch_info.append({
                        'x': new_w - patch_size,
                        'y': y,
                        'w': patch_size,
                        'h': patch_size,
                        'scale_x': self.original_size[0] / new_w,
                        'scale_y': self.original_size[1] / new_h
                    })
            
            if new_w % patch_size != 0 and new_h % patch_size != 0:
                patch = img_tensor[:, -patch_size:, -patch_size:]
                patches.append(patch)
                patch_info.append({
                    'x': new_w - patch_size,
                    'y': new_h - patch_size,
                    'w': patch_size,
                    'h': patch_size,
                    'scale_x': self.original_size[0] / new_w,
                    'scale_y': self.original_size[1] / new_h
                })
            
            print(f"Created {len(patches)} patches")
            return patches, patch_info
        else:
            # If image is smaller than target size, pad it
            pad_h = max(0, target_size - new_h)
            pad_w = max(0, target_size - new_w)
            padding = [pad_w//2, pad_h//2, pad_w-pad_w//2, pad_h-pad_h//2]
            img_tensor = F.pad(img_tensor, padding, mode='reflect')
            print("Image smaller than target size, padded to fit")
            return img_tensor, [{
                'x': 0,
                'y': 0,
                'w': new_w,
                'h': new_h,
                'scale_x': self.original_size[0] / new_w,
                'scale_y': self.original_size[1] / new_h
            }]
    
    def nms(self, boxes, scores, iou_threshold=0.3):
        """Apply Non-Maximum Suppression (NMS) to filter overlapping boxes."""
        if len(boxes) == 0:
            return []
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        keep = ops.nms(boxes_tensor, scores_tensor, iou_threshold)
        return keep.tolist()
    
    def predict(self, image_path, return_uncertainty=True, conf_threshold=0.5, nms_iou=0.3):
        """Make prediction on an image with NMS and confidence thresholding."""
        image_tensor, patch_info = self.preprocess_image(image_path)
        results = []
        if isinstance(image_tensor, list):  # Handle patches
            # Process each patch
            patch_predictions = []
            for patch, info in zip(image_tensor, patch_info):
                patch = patch.unsqueeze(0).to(self.device)
                with torch.no_grad():
                    pred = self.model(patch)
                patch_predictions.append((pred, info))
            # Get predictions for each patch
            for pred, info in patch_predictions:
                logits = pred['logits']
                probs = F.softmax(logits, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0, pred_class].item()
                if confidence > conf_threshold:  # Only include confident predictions
                    # Calculate patch coordinates in original image
                    x1 = int(info['x'] / info['scale_x'])
                    y1 = int(info['y'] / info['scale_y'])
                    x2 = int((info['x'] + info['w']) / info['scale_x'])
                    y2 = int((info['y'] + info['h']) / info['scale_y'])
                    results.append({
                        'class': self.class_names[pred_class],
                        'confidence': confidence,
                        'uncertainty': pred['uncertainty'].mean().item() if return_uncertainty else None,
                        'patch_info': info,
                        'logits': logits,
                        'bbox': [x1, y1, x2, y2],
                        'class_idx': pred_class
                    })
            # Apply NMS per class
            final_results = []
            for class_idx in set(r['class_idx'] for r in results):
                class_boxes = [r['bbox'] for r in results if r['class_idx'] == class_idx]
                class_scores = [r['confidence'] for r in results if r['class_idx'] == class_idx]
                class_results = [r for r in results if r['class_idx'] == class_idx]
                keep = self.nms(class_boxes, class_scores, iou_threshold=nms_iou)
                for idx in keep:
                    final_results.append(class_results[idx])
            # Sort by confidence
            final_results = sorted(final_results, key=lambda x: x['confidence'], reverse=True)
            return final_results
        else:
            # Process single image
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            with torch.no_grad():
                predictions = self.model(image_tensor)
                logits = predictions['logits']
                uncertainty = predictions['uncertainty']
            # Get probabilities
            probs = F.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()
            if confidence > conf_threshold:
                return [{
                    'class': self.class_names[pred_class],
                    'confidence': confidence,
                    'uncertainty': uncertainty.mean().item() if return_uncertainty else None,
                    'patch_info': patch_info[0],
                    'logits': logits,
                    'bbox': [0, 0, self.original_size[0], self.original_size[1]],
                    'class_idx': pred_class
                }]
            else:
                return []
    
    def _calculate_uncertainty(self, probs):
        """Calculate uncertainty using entropy."""
        return -torch.sum(probs * torch.log(probs + 1e-10)).item()
    
    def generate_cam(self, image_path):
        """Generate Class Activation Map for an image."""
        image_tensor = self.preprocess_image(image_path)
        if isinstance(image_tensor, list):
            # Use center patch for visualization
            mid_idx = len(image_tensor) // 2
            image_tensor = image_tensor[mid_idx]
        
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        return self.cam_generator.generate_cam(image_tensor)
    
    def visualize_cam(self, image_path, conf_threshold=0.5, nms_iou=0.3):
        """Generate and visualize CAM for all detected objects with NMS and thresholding."""
        # Load and preprocess image
        if isinstance(image_path, str):
            original_image = Image.open(image_path).convert('RGB')
        else:
            original_image = image_path
        # Get predictions for all patches
        predictions = self.predict(image_path, conf_threshold=conf_threshold, nms_iou=nms_iou)
        # Convert original image to numpy array
        img_np = np.array(original_image)
        # Create visualization
        import cv2
        visualization = img_np.copy()
        cam_info = []
        for pred in predictions:
            patch_info = pred['patch_info']
            bbox = pred['bbox']
            x1, y1, x2, y2 = bbox
            confidence = pred['confidence']
            color = (0, int(255 * confidence), 0)
            cv2.rectangle(visualization, (x1, y1), (x2, y2), color, 2)
            text = f"{pred['class']}: {confidence:.2f}"
            cv2.putText(visualization, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cam_info.append({
                'class': pred['class'],
                'classification_confidence': confidence,
                'localization_confidence': 0.0,  # Placeholder
                'combined_confidence': confidence,
                'bbox': {
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2
                }
            })
        return visualization, cam_info

def classify_image(
    image_path: str,
    model_path: Optional[str] = None,
    return_uncertainty: bool = True,
    max_image_size: int = 1024,
    patch_size: int = 224,
    overlap: float = 0.5
) -> Dict[str, Dict[str, any]]:
    """
    Convenience function for classifying a single image.
    
    Args:
        image_path (str): Path to the image file
        model_path (Optional[str]): Path to the model checkpoint
        return_uncertainty (bool): Whether to return uncertainty estimates
        max_image_size (int): Maximum size for large images
        patch_size (int): Size of patches for large images
        overlap (float): Overlap between patches
        
    Returns:
        Dict[str, Dict[str, any]]: Predictions for the image
    """
    predictor = AstroClassifierPredictor(
        model_path=model_path,
        max_image_size=max_image_size,
        patch_size=patch_size,
        overlap=overlap
    )
    return predictor.predict(image_path, return_uncertainty) 