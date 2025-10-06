"""
Neural Network Trainer for FluidNetSim.

Provides training capabilities for fluid dynamics prediction models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Callable
import logging
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

class FluidNetTrainer:
    """
    Trainer class for FluidNetSim neural networks.
    
    Features:
    - Multiple optimization algorithms
    - Learning rate scheduling
    - Early stopping
    - Model checkpointing
    - Training visualization
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: str = "adam",
        learning_rate: float = 1e-3,
        loss_function: str = "mse",
        device: str = "auto",
        **kwargs
    ):
        """
        Initialize trainer.
        
        Args:
            model: Neural network model to train
            optimizer: Optimization algorithm
            learning_rate: Initial learning rate
            loss_function: Loss function type
            device: Device to use for training
            **kwargs: Additional training parameters
        """
        self.model = model
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer(optimizer, learning_rate, **kwargs)
        
        # Setup loss function
        self.criterion = self._setup_loss_function(loss_function)
        
        # Training state
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.epoch = 0
        
        logger.info(f"Initialized trainer on device: {self.device}")
    
    def _setup_optimizer(self, optimizer: str, learning_rate: float, **kwargs) -> optim.Optimizer:
        """Setup optimization algorithm."""
        if optimizer.lower() == "adam":
            return optim.Adam(self.model.parameters(), lr=learning_rate, **kwargs)
        elif optimizer.lower() == "sgd":
            return optim.SGD(self.model.parameters(), lr=learning_rate, **kwargs)
        elif optimizer.lower() == "adamw":
            return optim.AdamW(self.model.parameters(), lr=learning_rate, **kwargs)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
    
    def _setup_loss_function(self, loss_function: str) -> nn.Module:
        """Setup loss function."""
        if loss_function.lower() == "mse":
            return nn.MSELoss()
        elif loss_function.lower() == "l1":
            return nn.L1Loss()
        elif loss_function.lower() == "huber":
            return nn.HuberLoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_function}")
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        **kwargs
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            epoch: Current epoch number
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            # Move data to device
            data = data.to(self.device)
            target = target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            
            # Compute loss
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'avg_loss': f'{total_loss/num_batches:.6f}'
            })
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return {
            'epoch': epoch,
            'train_loss': avg_loss,
            'num_batches': num_batches
        }
    
    def validate(
        self,
        dataloader: DataLoader,
        **kwargs
    ) -> Dict[str, float]:
        """
        Validate model on validation set.
        
        Args:
            dataloader: Validation data loader
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                # Move data to device
                data = data.to(self.device)
                target = target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                
                # Compute loss
                loss = self.criterion(output, target)
                
                # Update metrics
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        return {
            'val_loss': avg_loss,
            'num_batches': num_batches
        }
    
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        early_stopping_patience: int = 10,
        save_path: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            num_epochs: Number of training epochs
            early_stopping_patience: Patience for early stopping
            save_path: Path to save best model
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing training history
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(train_dataloader, epoch, **kwargs)
            
            # Validation
            if val_dataloader is not None:
                val_metrics = self.validate(val_dataloader, **kwargs)
                
                # Early stopping check
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    patience_counter = 0
                    
                    # Save best model
                    if save_path:
                        self.save_checkpoint(save_path, epoch, val_metrics)
                else:
                    patience_counter += 1
                
                # Log metrics
                logger.info(
                    f"Epoch {epoch}: "
                    f"Train Loss: {train_metrics['train_loss']:.6f}, "
                    f"Val Loss: {val_metrics['val_loss']:.6f}"
                )
                
                # Early stopping
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            else:
                logger.info(
                    f"Epoch {epoch}: "
                    f"Train Loss: {train_metrics['train_loss']:.6f}"
                )
        
        logger.info("Training completed")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss,
            'final_epoch': self.epoch
        }
    
    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        metrics: Dict[str, float]
    ) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'metrics': metrics
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.epoch = checkpoint['epoch']
        
        logger.info(f"Loaded checkpoint from {path}")
        return checkpoint
    
    def predict(
        self,
        dataloader: DataLoader,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate predictions.
        
        Args:
            dataloader: Data loader for prediction
            **kwargs: Additional parameters
            
        Returns:
            Model predictions
        """
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for data, _ in dataloader:
                data = data.to(self.device)
                output = self.model(data)
                predictions.append(output.cpu())
        
        return torch.cat(predictions, dim=0)
    
    def get_training_history(self) -> Dict[str, Any]:
        """Get training history."""
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': min(self.val_losses) if self.val_losses else None,
            'final_epoch': self.epoch
        }
    
    def __repr__(self) -> str:
        return f"FluidNetTrainer(model={type(self.model).__name__}, device={self.device})"
