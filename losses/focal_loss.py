import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import astro_config

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        # Class-specific gamma values
        self.gamma = {
            'asteroid': 2.0,
            'black_hole': 2.0,
            'comet': 2.0,
            'constellation': 3.0,  # Higher gamma for constellation class
            'galaxy': 2.0,
            'nebula': 2.0,
            'planet': 2.0,
            'star': 2.0
        }
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Get class-specific gamma values
        gamma_t = torch.tensor([self.gamma[name] for name in astro_config.config.class_names], 
                             device=targets.device)[targets]
        
        focal_loss = ((1 - pt) ** gamma_t * ce_loss)
        
        if self.alpha is not None:
            # Move alpha to the same device as targets
            alpha_t = self.alpha.to(targets.device)[targets]
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss 