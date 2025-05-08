"""
This module implements the training framework for the astronomical classifier using PyTorch Lightning.
It includes multi-task learning, uncertainty estimation, and comprehensive logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from pathlib import Path
import time
from datetime import timedelta
import GPUtil

class TrainingProgressCallback(pl.Callback):
    """
    Callback for tracking training progress and estimating completion time.
    """
    def __init__(self):
        super().__init__()
        self.start_time = None
        self.epoch_times = []
        self.batch_times = []
        self.gpu_utils = []
        self.memory_utils = []
        
    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()
        
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start = time.time()
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == 0:
            self.batch_start = time.time()
        elif batch_idx == 1:
            batch_time = time.time() - self.batch_start
            self.batch_times.append(batch_time)
            
            # Estimate time per epoch
            num_batches = len(trainer.train_dataloader)
            est_epoch_time = batch_time * num_batches
            
            # Get GPU utilization
            gpu = GPUtil.getGPUs()[0]
            self.gpu_utils.append(gpu.load * 100)
            self.memory_utils.append(gpu.memoryUtil * 100)
            
            # Calculate progress
            current_epoch = trainer.current_epoch
            total_epochs = trainer.max_epochs
            progress = (current_epoch + batch_idx / num_batches) / total_epochs
            
            # Estimate remaining time
            elapsed_time = time.time() - self.start_time
            est_total_time = elapsed_time / progress
            remaining_time = est_total_time - elapsed_time
            
            # Log progress
            pl_module.log('est_epoch_time', est_epoch_time)
            pl_module.log('est_remaining_time', remaining_time)
            pl_module.log('gpu_utilization', gpu.load * 100)
            pl_module.log('gpu_memory_util', gpu.memoryUtil * 100)
            
            # Print progress
            print(f"\nEpoch {current_epoch + 1}/{total_epochs}")
            print(f"Progress: {progress:.1%}")
            print(f"Est. time per epoch: {timedelta(seconds=int(est_epoch_time))}")
            print(f"Est. remaining time: {timedelta(seconds=int(remaining_time))}")
            print(f"GPU Utilization: {gpu.load * 100:.1f}%")
            print(f"GPU Memory: {gpu.memoryUtil * 100:.1f}%")
            
    def on_train_epoch_end(self, trainer, pl_module):
        epoch_time = time.time() - self.epoch_start
        self.epoch_times.append(epoch_time)
        
        # Calculate and log statistics
        avg_epoch_time = np.mean(self.epoch_times)
        avg_gpu_util = np.mean(self.gpu_utils)
        avg_memory_util = np.mean(self.memory_utils)
        
        pl_module.log('avg_epoch_time', avg_epoch_time)
        pl_module.log('avg_gpu_utilization', avg_gpu_util)
        pl_module.log('avg_memory_utilization', avg_memory_util)

class AstroClassifierTrainer(pl.LightningModule):
    """
    A PyTorch Lightning trainer for the astronomical classifier that handles multi-task learning.
    
    This class implements the training logic for the astronomical classifier, including
    multi-task learning with weighted losses, uncertainty estimation, and comprehensive
    logging of metrics. It uses PyTorch Lightning for efficient training and monitoring.
    
    Key features:
    - Multi-task learning with weighted losses
    - Uncertainty-aware training
    - Comprehensive metric logging
    - Learning rate scheduling
    - Early stopping and model checkpointing
    - Mixed precision training support
    
    Time Complexity: O(n) where n is the batch size
    Space Complexity: O(n) for batch processing
    """
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        task_weights: Dict[str, float] = None
    ):
        """
        Args:
            model (nn.Module): The AstroClassifier model
            learning_rate (float): Learning rate for optimization
            weight_decay (float): L2 regularization strength
            task_weights (Dict[str, float]): Weights for each classification task
        """
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Default task weights if not provided
        self.task_weights = task_weights or {
            'object': 1.0,
            'element': 0.8,
            'radiation': 0.8,
            'terrain': 0.6,
            'atmosphere': 0.6,
            'temperature': 0.7
        }
        
        # Loss functions for each task
        self.criterion = nn.CrossEntropyLoss()
        
        # Performance tracking
        self.training_start_time = None
        self.epoch_times = []
        
    def forward(self, x: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input images
            
        Returns:
            Dict[str, Dict[str, torch.Tensor]]: Model predictions
        """
        return self.model(x)
        
    def training_step(self, batch: Tuple[torch.Tensor, Dict[str, torch.Tensor]], batch_idx: int) -> torch.Tensor:
        """
        Training step with multi-task learning.
        
        Args:
            batch (Tuple[torch.Tensor, Dict[str, torch.Tensor]]): Input batch and labels
            batch_idx (int): Batch index
            
        Returns:
            torch.Tensor: Total loss
        """
        x, y = batch
        predictions = self(x)
        
        # Calculate loss for each task
        total_loss = 0
        for task_name, task_pred in predictions.items():
            task_loss = self.criterion(task_pred['logits'], y[task_name])
            total_loss += self.task_weights[task_name] * task_loss
            
        # Log losses
        self.log('train_loss', total_loss, prog_bar=True)
        for task_name, task_pred in predictions.items():
            self.log(f'train_{task_name}_loss', 
                    self.criterion(task_pred['logits'], y[task_name]),
                    prog_bar=True)
            
        return total_loss
        
    def validation_step(self, batch: Tuple[torch.Tensor, Dict[str, torch.Tensor]], batch_idx: int) -> None:
        """
        Validation step with uncertainty estimation.
        
        Args:
            batch (Tuple[torch.Tensor, Dict[str, torch.Tensor]]): Input batch and labels
            batch_idx (int): Batch index
        """
        x, y = batch
        predictions = self(x)
        
        # Calculate validation metrics
        total_loss = 0
        for task_name, task_pred in predictions.items():
            task_loss = self.criterion(task_pred['logits'], y[task_name])
            total_loss += self.task_weights[task_name] * task_loss
            
            # Calculate accuracy
            pred_labels = torch.argmax(task_pred['logits'], dim=1)
            accuracy = (pred_labels == y[task_name]).float().mean()
            
            # Log metrics
            self.log(f'val_{task_name}_loss', task_loss)
            self.log(f'val_{task_name}_accuracy', accuracy)
            
        self.log('val_loss', total_loss)
        
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure optimizer with learning rate scheduling.
        
        Returns:
            torch.optim.Optimizer: Optimizer instance
        """
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler with warmup
        scheduler = {
            'scheduler': optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=len(self.trainer.train_dataloader),
                pct_start=0.1,  # 10% warmup
                div_factor=25,  # initial_lr = max_lr/25
                final_div_factor=1e4
            ),
            'interval': 'step'
        }
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        
    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Prediction step with uncertainty estimation.
        
        Args:
            batch (torch.Tensor): Input batch
            batch_idx (int): Batch index
            
        Returns:
            Dict[str, Dict[str, torch.Tensor]]: Predictions with uncertainties
        """
        predictions = self(batch)
        
        # Add uncertainty information to predictions
        for task_name, task_pred in predictions.items():
            task_pred['uncertainty_score'] = task_pred['uncertainty'].mean(dim=1)
            
        return predictions

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    max_epochs: int = 100,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    task_weights: Dict[str, float] = None,
    checkpoint_dir: str = 'checkpoints'
) -> AstroClassifierTrainer:
    """
    Train the astronomical classifier model with progress tracking and performance monitoring.
    
    Args:
        model (nn.Module): The AstroClassifier model
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        max_epochs (int): Maximum number of training epochs
        learning_rate (float): Learning rate for optimization
        weight_decay (float): L2 regularization strength
        task_weights (Dict[str, float]): Weights for each classification task
        checkpoint_dir (str): Directory to save model checkpoints
        
    Returns:
        AstroClassifierTrainer: Trained model trainer
    """
    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize trainer
    trainer = AstroClassifierTrainer(
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        task_weights=task_weights
    )
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='astro-classifier-{epoch:02d}-{val_loss:.2f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min'
        ),
        TrainingProgressCallback()  # Add progress tracking
    ]
    
    # Initialize PyTorch Lightning trainer
    pl_trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        accelerator='auto',
        devices='auto',
        precision=16,  # Use mixed precision training
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        val_check_interval=0.25  # Validate every 25% of training
    )
    
    # Train the model
    pl_trainer.fit(
        trainer,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
    
    return trainer 