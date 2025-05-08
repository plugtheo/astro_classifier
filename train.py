"""
Main training script for the astronomical classifier.
"""

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                       ModelCheckpoint)
from pytorch_lightning.loggers import TensorBoardLogger

from config.config import astro_config
from data.dataset import create_dataloaders
from models.astro_classifier import AstroClassifier
from models.astro_classifier_module import AstroClassifierModule
from utils.training_utils import print_gpu_memory, verify_gradients
from utils.model_utils import save_model_checkpoint

def main():
    # Clear GPU memory before training
    torch.cuda.empty_cache()

    # Set environment variables for better memory management
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Enable Tensor Cores for better performance
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        torch.cuda.empty_cache()
    
    # Set random seed for reproducibility
    pl.seed_everything(42)
    
    # Create data loaders
    print("\nCreating data loaders...")
    dataloaders = create_dataloaders(
        data_dir=astro_config.config.data_dir,
        batch_size=astro_config.config.batch_size,
        num_workers=astro_config.config.num_workers,
        max_samples=None,  # Use all available samples
        persistent_workers=astro_config.config.persistent_workers
    )
    print("Data loaders created.")
    
    # Initialize model
    print("\nInitializing model...")
    model = AstroClassifier(
        num_classes=astro_config.config.num_classes,
        backbone=astro_config.config.backbone,
        pretrained=astro_config.config.pretrained
    )
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Ensure all parameters require gradients
    model.requires_grad_(True)
    
    # Double check and force gradients for all parameters
    for param in model.parameters():
        param.requires_grad_(True)
    
    # Verify gradients are enabled
    if not verify_gradients(model):
        raise RuntimeError("Model parameters do not have gradients enabled!")
    
    print("Model initialized.")
    
    # Wrap model in Lightning module
    model_module = AstroClassifierModule(model)
    
    # Initialize trainer with memory-efficient settings
    trainer = pl.Trainer(
        max_epochs=astro_config.config.max_epochs,
        accelerator=astro_config.config.accelerator,
        devices=astro_config.config.devices,
        precision=astro_config.config.precision,
        accumulate_grad_batches=astro_config.config.gradient_accumulation_steps,
        gradient_clip_val=astro_config.config.gradient_clip_val,
        logger=TensorBoardLogger(astro_config.config.log_dir, name='astro_classifier'),
        callbacks=[
            ModelCheckpoint(
                dirpath=astro_config.config.checkpoint_dir,
                filename='astro-classifier-{epoch:02d}-{val_loss:.2f}',
                monitor='val_loss',
                mode='min',
                save_top_k=astro_config.config.save_top_k,
                save_weights_only=True  # Only save model weights
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=astro_config.config.early_stopping_patience,
                mode='min'
            ),
            LearningRateMonitor(logging_interval='step')
        ],
        val_check_interval=astro_config.config.val_check_interval,
        enable_progress_bar=True,
        enable_model_summary=True,
        enable_checkpointing=True,  # Explicitly enable checkpointing
        gradient_clip_algorithm='norm'  # Use gradient norm clipping
    )
    
    # Train model
    print("\nStarting training...")
    trainer.fit(
        model_module,
        train_dataloaders=dataloaders['train'],
        val_dataloaders=dataloaders['val']
    )
    
    # Test model
    print("\nRunning final test...")
    trainer.test(model_module, dataloaders=dataloaders['test'])
    
    # Save final model
    val_loss = trainer.callback_metrics.get('val_loss', torch.tensor(float('inf')))
    save_model_checkpoint(
        model=model_module.model,
        save_dir=astro_config.config.checkpoint_dir,
        epoch=trainer.current_epoch,
        val_loss=val_loss.item() if isinstance(val_loss, torch.Tensor) else float('inf')
    )
    
    # Clear CUDA cache after training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main() 