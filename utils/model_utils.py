"""
Utility functions for model management, including saving, loading, and checkpoint handling.
"""

import os
import torch
import json
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
import numpy as np

from config.config import astro_config

model_config = astro_config.config

def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Get the path to the latest checkpoint file.
    
    Args:
        checkpoint_dir (str): Directory containing checkpoints
        
    Returns:
        Optional[str]: Path to latest checkpoint if exists, None otherwise
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
        
    checkpoints = list(checkpoint_dir.glob('*.ckpt'))
    if not checkpoints:
        return None
        
    return str(max(checkpoints, key=lambda x: x.stat().st_mtime))

def load_pretrained_model(model: torch.nn.Module, checkpoint_path: str) -> torch.nn.Module:
    """
    Load pretrained weights into a model.
    
    Args:
        model (torch.nn.Module): Model to load weights into
        checkpoint_path (str): Path to checkpoint file
        
    Returns:
        torch.nn.Module: Model with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    return model

def save_model_checkpoint(
    model: torch.nn.Module,
    save_dir: str,
    epoch: int,
    val_loss: float,
    additional_info: Optional[Dict] = None
) -> None:
    """
    Save model checkpoint with metadata.
    
    Args:
        model (torch.nn.Module): Model to save
        save_dir (str): Directory to save checkpoint
        epoch (int): Current epoch
        val_loss (float): Validation loss
        additional_info (Optional[Dict]): Additional information to save
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare metadata
    metadata = {
        'epoch': epoch,
        'val_loss': float(val_loss),  # Ensure val_loss is a Python float
        'timestamp': datetime.now().isoformat(),
        'model_architecture': model.__class__.__name__
    }
    
    # Save model weights
    checkpoint_path = os.path.join(save_dir, f'model_epoch_{epoch}.pth')
    torch.save(model.state_dict(), checkpoint_path)
    
    # Save metadata
    metadata_path = os.path.join(save_dir, f'model_epoch_{epoch}_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2) 