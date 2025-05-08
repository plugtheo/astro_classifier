"""
This module implements the core model architecture for astronomical object classification.
It includes uncertainty estimation, multi-task learning, and a reusable discriminator.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

from config.config import astro_config


class UncertaintyEstimator(nn.Module):
    """
    A Monte Carlo Dropout-based uncertainty estimator that provides both predictions and uncertainty measures.
    
    This class implements Bayesian uncertainty estimation by performing multiple forward passes
    with dropout enabled during inference. It calculates both the mean prediction and the
    standard deviation across multiple samples to quantify prediction uncertainty.
    
    Key features:
    - Enables dropout during inference for uncertainty estimation
    - Performs multiple forward passes (Monte Carlo sampling)
    - Calculates mean predictions and uncertainty measures
    - Helps identify when the model is uncertain about its predictions
    
    Time Complexity: O(1) for forward pass
    Space Complexity: O(1) for model parameters
    """
    def __init__(self, dropout_rate: float = 0.2):
        """
        Args:
            dropout_rate (float): Probability of dropping neurons during inference
                                 Higher values increase uncertainty estimation
                                 but may reduce model confidence
        """
        super().__init__()
        self.dropout_rate = dropout_rate
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with Monte Carlo Dropout.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean predictions and uncertainty estimates
        """
        # Enable dropout during inference
        self.train()
        predictions = []
        
        # Multiple forward passes for uncertainty estimation
        for _ in range(10):  # Number of Monte Carlo samples
            pred = F.dropout(x, p=self.dropout_rate, training=True)
            predictions.append(pred)
            
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)
        
        return mean_pred, uncertainty

class MultiTaskHead(nn.Module):
    """
    A multi-task classification head that combines classification with uncertainty estimation.
    
    This class implements a classification head that can be used for different tasks
    (e.g., object classification, element detection) while providing uncertainty estimates
    for each prediction. It includes a classifier network and an uncertainty estimator.
    
    Key features:
    - Flexible classification head for different tasks
    - Integrated uncertainty estimation
    - Dropout for regularization and uncertainty
    - Softmax output for probability distributions
    - Feature refinement layers for better task-specific learning
    
    Time Complexity: O(n) where n is the number of classes
    Space Complexity: O(n) for classification layer parameters
    """
    def __init__(self, in_features: int, num_classes: int, dropout_rate: float = 0.2):
        """
        Args:
            in_features (int): Number of input features
            num_classes (int): Number of output classes
            dropout_rate (float): Dropout rate for uncertainty estimation
        """
        super().__init__()
        # Feature refinement layers
        self.feature_refinement = nn.Sequential(
            nn.Linear(in_features, 512),  # Reduced from 1024 to 512 for ResNet18
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),  # Reduced from 512 to 256 for ResNet18
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Classification layer - ensure output matches num_classes
        self.classifier = nn.Linear(256, num_classes)  # Changed from 512 to 256
        self.uncertainty = UncertaintyEstimator(dropout_rate)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with uncertainty estimation.
        
        Args:
            x (torch.Tensor): Input features
            
        Returns:
            Dict[str, torch.Tensor]: Predictions and uncertainties
        """
        # Apply feature refinement
        refined_features = self.feature_refinement(x)
        
        # Get logits - this will now output num_classes dimensions
        logits = self.classifier(refined_features)
        
        # Get uncertainty estimates
        mean_pred, uncertainty = self.uncertainty(logits)  # Changed to use logits instead of refined_features
        
        return {
            'logits': mean_pred,  # This will be num_classes dimensions
            'uncertainty': uncertainty,
            'probabilities': F.softmax(mean_pred, dim=1),
            'features': refined_features  # Return refined features for potential use
        }

class AstroClassifier(nn.Module):
    """
    The main astronomical classifier model that combines multiple classification tasks.
    
    This class implements a multi-task learning model for astronomical object classification,
    combining a shared feature extractor (ResNet18) with multiple task-specific heads.
    It can classify astronomical objects into 8 categories: asteroid, black hole, comet,
    constellation, galaxy, nebula, planet, and star.
    
    Key features:
    - Transfer learning with pretrained ResNet18 backbone
    - Classification head with uncertainty estimation
    - Feature refinement for better task-specific learning
    - Reusable discriminator for GAN applications
    - Flexible architecture supporting different backbones
    
    Time Complexity: O(n) where n is the number of layers
    Space Complexity: O(n) for model parameters
    """
    def __init__(
        self,
        num_classes: int = 8,
        backbone: str = 'resnet18',
        pretrained: bool = True
    ):
        """
        Args:
            num_classes (int): Number of astronomical object classes (default: 8)
            backbone (str): CNN backbone architecture
            pretrained (bool): Whether to use pretrained weights
        """
        super().__init__()
        
        # Load backbone
        if backbone == 'resnet18':
            self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Feature refinement with attention
        self.feature_refinement = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Class-specific attention layers
        self.class_attention = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.Sigmoid()
            ) for name in astro_config.config.class_names
        })
        
        # Final classification layer
        self.classifier = nn.Linear(512, num_classes)
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract features
        features = self.backbone(x)
        
        # Refine features
        refined_features = self.feature_refinement(features)
        
        # Apply class-specific attention
        attended_features = []
        for class_name in astro_config.config.class_names:
            attention = self.class_attention[class_name](refined_features)
            attended = refined_features * attention
            attended_features.append(attended)
        
        # Combine attended features
        combined_features = torch.stack(attended_features).mean(dim=0)
        
        # Classification
        logits = self.classifier(combined_features)
        
        # Uncertainty estimation
        uncertainty = self.uncertainty_head(combined_features)
        
        return {
            'logits': logits,
            'uncertainty': uncertainty.squeeze(-1)
        }
        
    def get_discriminator(self) -> nn.Module:
        """
        Returns a reusable discriminator architecture for GAN applications.
        
        Returns:
            nn.Module: Discriminator model
        """
        class Discriminator(nn.Module):
            def __init__(self, in_features: int):
                super().__init__()
                self.model = nn.Sequential(
                    nn.Linear(in_features, 512),  # Reduced from 1024 to 512
                    nn.LeakyReLU(0.2),
                    nn.BatchNorm1d(512),
                    nn.Dropout(0.3),
                    nn.Linear(512, 256),  # Reduced from 512 to 256
                    nn.LeakyReLU(0.2),
                    nn.BatchNorm1d(256),
                    nn.Dropout(0.3),
                    nn.Linear(256, 1),
                    nn.Sigmoid()
                )
                
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.model(x)
                
        return Discriminator(512)  # Changed from 1280 to 512 for ResNet18 