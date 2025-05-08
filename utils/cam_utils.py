import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict

class ClassActivationMap:
    """
    Generates Class Activation Maps (CAM) for visualizing which regions of the input image
    contribute to the model's classification decision.
    """
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        self.features = None
        self.gradients = None
        
        # Register hooks to get feature maps and gradients
        self.model.backbone.layer4.register_forward_hook(self._get_features)
        self.model.backbone.layer4.register_full_backward_hook(self._get_gradients)
    
    def _get_features(self, module, input, output):
        """Hook to get feature maps from the last layer"""
        self.features = output.detach()
    
    def _get_gradients(self, module, grad_input, grad_output):
        """Hook to get gradients from the last layer"""
        self.gradients = grad_output[0].detach()
    
    def _calculate_localization_confidence(self, cam: np.ndarray, bbox: Dict) -> float:
        """
        Calculate confidence score for the localization based on CAM intensity.
        
        Args:
            cam: Class Activation Map
            bbox: Bounding box coordinates
            
        Returns:
            float: Localization confidence score [0, 1]
        """
        if bbox is None:
            return 0.0
            
        # Extract CAM region within the bounding box
        x1, y1 = bbox['x1'], bbox['y1']
        x2, y2 = bbox['x2'], bbox['y2']
        
        # Scale coordinates to CAM size
        h, w = cam.shape
        x1_cam = int(x1 * w / self.features.shape[3])
        y1_cam = int(y1 * h / self.features.shape[2])
        x2_cam = int(x2 * w / self.features.shape[3])
        y2_cam = int(y2 * h / self.features.shape[2])
        
        # Get CAM values within the box
        box_cam = cam[y1_cam:y2_cam, x1_cam:x2_cam]
        
        if box_cam.size == 0:
            return 0.0
            
        # Calculate confidence based on:
        # 1. Mean intensity within the box
        # 2. Ratio of high-intensity pixels
        mean_intensity = np.mean(box_cam)
        high_intensity_ratio = np.mean(box_cam > 0.5)
        
        # Combine scores (you can adjust weights)
        confidence = 0.7 * mean_intensity + 0.3 * high_intensity_ratio
        
        return float(confidence)
    
    def generate_cam(self, input_tensor: torch.Tensor, class_idx: int = None) -> Tuple[np.ndarray, Dict]:
        """
        Generate Class Activation Map for the input image.
        
        Args:
            input_tensor: Input image tensor
            class_idx: Index of the class to generate CAM for. If None, uses the predicted class.
            
        Returns:
            Tuple containing:
            - CAM heatmap as numpy array
            - Dictionary with additional information (predicted class, confidence, etc.)
        """
        # Ensure input is on the correct device
        input_tensor = input_tensor.to(self.device)
        
        # Get model predictions
        with torch.set_grad_enabled(True):
            output = self.model(input_tensor)
            logits = output['logits']
            
            if class_idx is None:
                # Use the predicted class
                class_idx = logits.argmax(dim=1).item()
            
            # Get the score for the target class
            score = logits[0, class_idx]
            
            # Backward pass to get gradients
            self.model.zero_grad()
            score.backward()
        
        # Get the weights for the target class
        weights = torch.mean(self.gradients, dim=(2, 3))
        
        # Generate CAM
        cam = torch.zeros(self.features.shape[2:], dtype=torch.float32, device=self.device)
        for i, w in enumerate(weights[0]):
            cam += w * self.features[0, i, :, :]
        
        # Apply ReLU to get only positive contributions
        cam = F.relu(cam)
        
        # Normalize CAM
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        
        # Convert to numpy and resize to input image size
        cam = cam.cpu().numpy()
        
        # Get bounding box from CAM
        threshold = 0.5  # You can adjust this threshold
        binary_cam = (cam > threshold).astype(np.uint8)
        
        # Find contours and get bounding box
        from skimage import measure
        contours = measure.find_contours(binary_cam, 0.5)
        
        if len(contours) > 0:
            # Get the largest contour
            contour = max(contours, key=lambda x: len(x))
            y_min, x_min = contour.min(axis=0)
            y_max, x_max = contour.max(axis=0)
            
            # Scale coordinates to original image size
            h, w = input_tensor.shape[2:]
            bbox = {
                'x1': int(x_min * w / cam.shape[1]),
                'y1': int(y_min * h / cam.shape[0]),
                'x2': int(x_max * w / cam.shape[1]),
                'y2': int(y_max * h / cam.shape[0])
            }
        else:
            bbox = None
        
        # Calculate confidence scores
        classification_confidence = F.softmax(logits, dim=1)[0, class_idx].item()
        localization_confidence = self._calculate_localization_confidence(cam, bbox)
        
        # Combined confidence score
        combined_confidence = 0.6 * classification_confidence + 0.4 * localization_confidence
        
        return cam, {
            'class_idx': class_idx,
            'classification_confidence': classification_confidence,
            'localization_confidence': localization_confidence,
            'combined_confidence': combined_confidence,
            'bbox': bbox
        }
    
    def visualize_cam(self, input_tensor: torch.Tensor, class_idx: int = None) -> Tuple[np.ndarray, Dict]:
        """
        Generate and visualize CAM with bounding box.
        
        Args:
            input_tensor: Input image tensor
            class_idx: Index of the class to generate CAM for. If None, uses the predicted class.
            
        Returns:
            Tuple containing:
            - Visualization image as numpy array
            - Dictionary with additional information
        """
        import cv2
        
        # Generate CAM
        cam, info = self.generate_cam(input_tensor, class_idx)
        
        # Convert input tensor to image
        img = input_tensor[0].cpu().numpy().transpose(1, 2, 0)
        # Denormalize image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Resize heatmap to match image size
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        
        # Blend heatmap with original image
        alpha = 0.6
        visualization = cv2.addWeighted(img, 1-alpha, heatmap, alpha, 0)
        
        # Draw bounding box if available
        if info['bbox'] is not None:
            bbox = info['bbox']
            # Draw box with confidence-based color
            confidence = info['combined_confidence']
            color = (0, int(255 * confidence), 0)  # Green intensity based on confidence
            cv2.rectangle(visualization, 
                         (bbox['x1'], bbox['y1']), 
                         (bbox['x2'], bbox['y2']), 
                         color, 2)
            
            # Add confidence scores as text
            text = f"Conf: {info['combined_confidence']:.2f}"
            cv2.putText(visualization, text, 
                       (bbox['x1'], bbox['y1'] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return visualization, info 