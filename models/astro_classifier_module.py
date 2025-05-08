import json
from pathlib import Path
import torch
import torchmetrics
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import OneCycleLR

from config.config import astro_config
from losses.focal_loss import FocalLoss

class AstroClassifierModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Convert class weights dict to tensor
        weights = [astro_config.config.class_weights[name] for name in astro_config.config.class_names]
        self.criterion = FocalLoss(alpha=torch.tensor(weights), gamma=2.0)
        
        # Initialize metrics for test phase
        self.test_metrics = {
            'accuracy': torchmetrics.Accuracy(task='multiclass', num_classes=len(astro_config.config.class_names)),
            'f1': torchmetrics.F1Score(task='multiclass', num_classes=len(astro_config.config.class_names)),
            'precision': torchmetrics.Precision(task='multiclass', num_classes=len(astro_config.config.class_names)),
            'recall': torchmetrics.Recall(task='multiclass', num_classes=len(astro_config.config.class_names)),
            'confusion_matrix': torchmetrics.ConfusionMatrix(task='multiclass', num_classes=len(astro_config.config.class_names))
        }
        
        # Track epoch-level metrics
        self.epoch_metrics = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
    
    def setup(self, stage=None):
        """Move metrics to the correct device."""
        if stage == 'test':
            for metric in self.test_metrics.values():
                metric.to(self.device)
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch

        # Move to GPU efficiently
        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        outputs = self(images)
        
        logits = outputs['logits']
        uncertainty = outputs['uncertainty']
        loss = self.criterion(logits, labels)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_uncertainty', torch.mean(uncertainty), prog_bar=True, on_step=True, on_epoch=True)
        self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        logits = outputs['logits']
        uncertainty = outputs['uncertainty']
        loss = self.criterion(logits, labels)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).float().mean()
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_accuracy', accuracy, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_uncertainty', torch.mean(uncertainty), prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def on_train_epoch_end(self):
        # Store epoch metrics
        self.epoch_metrics['train_loss'].append(self.trainer.callback_metrics['train_loss'].item())
        
        # Print epoch summary
        print(f"\nEpoch {self.current_epoch} Summary:")
        print(f"Training Loss: {self.epoch_metrics['train_loss'][-1]:.4f}")
        if 'val_loss' in self.trainer.callback_metrics:
            print(f"Validation Loss: {self.trainer.callback_metrics['val_loss'].item():.4f}")
            print(f"Validation Accuracy: {self.trainer.callback_metrics['val_accuracy'].item():.4f}")
    
    def on_validation_epoch_end(self):
        # Store validation metrics
        if 'val_loss' in self.trainer.callback_metrics:
            self.epoch_metrics['val_loss'].append(self.trainer.callback_metrics['val_loss'].item())
            self.epoch_metrics['val_accuracy'].append(self.trainer.callback_metrics['val_accuracy'].item())
    
    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        logits = outputs['logits']
        uncertainty = outputs['uncertainty']
        loss = self.criterion(logits, labels)
        
        # Get predictions
        preds = torch.argmax(logits, dim=1)
        
        # Update metrics
        for metric_name, metric in self.test_metrics.items():
            if metric_name == 'confusion_matrix':
                metric.update(preds, labels)
            else:
                metric.update(logits, labels)
        
        # Calculate per-class metrics
        for i, class_name in enumerate(astro_config.config.class_names):
            class_mask = (labels == i)
            if class_mask.any():
                class_acc = (preds[class_mask] == labels[class_mask]).float().mean()
                self.log(f'test_{class_name}_accuracy', class_acc)
        
        # Log overall metrics
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_uncertainty', torch.mean(uncertainty), prog_bar=True)
        
        return {
            'test_loss': loss,
            'test_uncertainty': torch.mean(uncertainty),
            'preds': preds,
            'labels': labels
        }
    
    def on_test_epoch_end(self):
        # Log final test metrics
        for metric_name, metric in self.test_metrics.items():
            if metric_name == 'confusion_matrix':
                cm = metric.compute()
                # Log confusion matrix as a figure
                fig = plt.figure(figsize=(10, 10))
                sns.heatmap(cm.cpu().numpy(), annot=True, fmt='d', cmap='Blues',
                           xticklabels=astro_config.config.class_names,
                           yticklabels=astro_config.config.class_names)
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                self.logger.experiment.add_figure('test_confusion_matrix', fig, self.current_epoch)
            else:
                self.log(f'test_{metric_name}', metric.compute())
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=astro_config.config.learning_rate,
            weight_decay=astro_config.config.weight_decay
        )
        
        # Get total number of valid images from training metadata
        metadata_path = Path(astro_config.config.data_dir) / 'metadata' / 'train_metadata.json'
        with open(metadata_path, 'r') as f:
            train_metadata = json.load(f)
        total_valid_images = len(train_metadata)
        
        # Calculate steps per epoch based on config values
        # Effective batch size = batch_size * gradient_accumulation_steps
        effective_batch_size = astro_config.config.batch_size * astro_config.config.gradient_accumulation_steps
        steps_per_epoch = total_valid_images // effective_batch_size  # Integer division to get number of steps
        
        # Learning rate scheduler with warmup
        scheduler = {
            'scheduler': OneCycleLR(
                optimizer,
                max_lr=3e-4,  # Increased from 1e-4
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.3,  # Longer warmup
                div_factor=10,  # Less aggressive initial LR reduction
                final_div_factor=1e4
            ),
            'interval': 'step'
        }
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler} 