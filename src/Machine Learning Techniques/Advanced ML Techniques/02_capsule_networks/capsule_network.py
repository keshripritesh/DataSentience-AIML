"""
Complete Capsule Network Architecture
This module implements the full CapsNet architecture as described in the original paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List

from capsule_layer import CapsuleLayer, PrimaryCapsuleLayer, ReconstructionLayer, compute_capsule_accuracy


class CapsNet(nn.Module):
    """
    Complete Capsule Network architecture.
    
    Architecture:
    1. Convolutional layer (feature extraction)
    2. Primary capsule layer (convert features to capsules)
    3. Digit capsule layer (class capsules with routing)
    4. Reconstruction layer (optional, for regularization)
    """
    
    def __init__(self,
                 input_channels: int = 1,
                 conv_channels: int = 256,
                 primary_capsules: int = 32,
                 primary_capsule_dim: int = 8,
                 digit_capsules: int = 10,
                 digit_capsule_dim: int = 16,
                 num_routing_iterations: int = 3,
                 use_reconstruction: bool = True,
                 reconstruction_weight: float = 0.0005):
        super(CapsNet, self).__init__()
        
        self.input_channels = input_channels
        self.conv_channels = conv_channels
        self.primary_capsules = primary_capsules
        self.primary_capsule_dim = primary_capsule_dim
        self.digit_capsules = digit_capsules
        self.digit_capsule_dim = digit_capsule_dim
        self.num_routing_iterations = num_routing_iterations
        self.use_reconstruction = use_reconstruction
        self.reconstruction_weight = reconstruction_weight
        
        # Convolutional layer for feature extraction
        self.conv1 = nn.Conv2d(
            input_channels, 
            conv_channels, 
            kernel_size=9, 
            stride=1, 
            padding=0
        )
        
        # Primary capsule layer
        self.primary_capsules = PrimaryCapsuleLayer(
            in_channels=conv_channels,
            num_capsules=primary_capsules,
            capsule_dim=primary_capsule_dim,
            kernel_size=9,
            stride=2
        )
        
        # Calculate number of primary capsules after convolution
        # Assuming input size of 28x28 (MNIST)
        # After conv1: 20x20, after primary capsules: 6x6
        self.num_primary_capsules = primary_capsules * 6 * 6
        
        # Digit capsule layer
        self.digit_capsules = CapsuleLayer(
            num_in_capsules=self.num_primary_capsules,
            in_capsule_dim=primary_capsule_dim,
            num_out_capsules=digit_capsules,
            out_capsule_dim=digit_capsule_dim,
            num_routing_iterations=num_routing_iterations
        )
        
        # Reconstruction layer (optional)
        if use_reconstruction:
            # Calculate input dimension for reconstruction
            # Assuming MNIST: 28 * 28 = 784
            self.reconstruction = ReconstructionLayer(
                num_capsules=digit_capsules,
                capsule_dim=digit_capsule_dim,
                input_dim=784,  # 28x28 for MNIST
                hidden_dim=512
            )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the CapsNet.
        
        Args:
            x: Input images [batch_size, channels, height, width]
            
        Returns:
            Dictionary containing:
            - digit_capsules: Output digit capsules
            - reconstructed: Reconstructed images (if reconstruction enabled)
            - activities: Capsule activities (magnitudes)
        """
        batch_size = x.size(0)
        
        # Convolutional feature extraction
        # Shape: [batch_size, conv_channels, height, width]
        conv_features = F.relu(self.conv1(x))
        
        # Primary capsules
        # Shape: [batch_size, num_primary_capsules, primary_capsule_dim]
        primary_caps = self.primary_capsules(conv_features)
        
        # Digit capsules with routing
        # Shape: [batch_size, digit_capsules, digit_capsule_dim]
        digit_caps = self.digit_capsules(primary_caps)
        
        # Compute capsule activities (magnitudes)
        # Shape: [batch_size, digit_capsules]
        activities = torch.norm(digit_caps, dim=-1)
        
        # Reconstruction (optional)
        reconstructed = None
        if self.use_reconstruction:
            reconstructed = self.reconstruction(digit_caps, x)
        
        return {
            'digit_capsules': digit_caps,
            'reconstructed': reconstructed,
            'activities': activities,
            'primary_capsules': primary_caps
        }
    
    def compute_loss(self, 
                    digit_capsules: torch.Tensor,
                    targets: torch.Tensor,
                    reconstructed: Optional[torch.Tensor] = None,
                    original_images: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute the total loss including margin loss and reconstruction loss.
        
        Args:
            digit_capsules: Output digit capsules [batch_size, num_classes, capsule_dim]
            targets: Target labels [batch_size]
            reconstructed: Reconstructed images [batch_size, channels, height, width]
            original_images: Original input images [batch_size, channels, height, width]
            
        Returns:
            Dictionary containing total loss and component losses
        """
        # Margin loss for classification
        margin_loss = self._margin_loss(digit_capsules, targets)
        
        # Reconstruction loss (if enabled)
        reconstruction_loss = torch.tensor(0.0, device=digit_capsules.device)
        if reconstructed is not None and original_images is not None:
            reconstruction_loss = F.mse_loss(reconstructed, original_images, reduction='sum')
            reconstruction_loss = reconstruction_loss / original_images.size(0)  # Average over batch
        
        # Total loss
        total_loss = margin_loss + self.reconstruction_weight * reconstruction_loss
        
        return {
            'total_loss': total_loss,
            'margin_loss': margin_loss,
            'reconstruction_loss': reconstruction_loss
        }
    
    def _margin_loss(self, digit_capsules: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute margin loss for capsule classification.
        
        Args:
            digit_capsules: Output digit capsules [batch_size, num_classes, capsule_dim]
            targets: Target labels [batch_size]
            
        Returns:
            Margin loss
        """
        batch_size = digit_capsules.size(0)
        num_classes = digit_capsules.size(1)
        
        # Compute capsule activities (magnitudes)
        # Shape: [batch_size, num_classes]
        activities = torch.norm(digit_capsules, dim=-1)
        
        # Create one-hot encoded targets
        # Shape: [batch_size, num_classes]
        targets_one_hot = F.one_hot(targets, num_classes).float()
        
        # Margin loss parameters
        m_plus = 0.9
        m_minus = 0.1
        lambda_val = 0.5
        
        # Loss for correct class (should be large)
        # Shape: [batch_size, num_classes]
        left = F.relu(m_plus - activities, inplace=True) ** 2
        left = left * targets_one_hot
        
        # Loss for incorrect classes (should be small)
        # Shape: [batch_size, num_classes]
        right = F.relu(activities - m_minus, inplace=True) ** 2
        right = right * (1.0 - targets_one_hot)
        
        # Combine losses
        margin_loss = left + lambda_val * right
        margin_loss = margin_loss.sum(dim=1).mean()
        
        return margin_loss
    
    def predict(self, digit_capsules: torch.Tensor) -> torch.Tensor:
        """
        Make predictions based on capsule activities.
        
        Args:
            digit_capsules: Output digit capsules [batch_size, num_classes, capsule_dim]
            
        Returns:
            Predicted class labels [batch_size]
        """
        # Compute capsule activities (magnitudes)
        activities = torch.norm(digit_capsules, dim=-1)
        
        # Predict class with maximum activity
        predictions = torch.argmax(activities, dim=1)
        
        return predictions
    
    def get_capsule_visualization(self, digit_capsules: torch.Tensor) -> torch.Tensor:
        """
        Get capsule activities for visualization.
        
        Args:
            digit_capsules: Output digit capsules [batch_size, num_classes, capsule_dim]
            
        Returns:
            Capsule activities [batch_size, num_classes]
        """
        return torch.norm(digit_capsules, dim=-1)


class CapsNetTrainer:
    """
    Trainer for Capsule Networks.
    """
    
    def __init__(self, 
                 model: CapsNet,
                 train_loader,
                 val_loader,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 0.001):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_margin_loss = 0
        total_reconstruction_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(data)
            digit_capsules = outputs['digit_capsules']
            reconstructed = outputs.get('reconstructed')
            
            # Compute loss
            loss_dict = self.model.compute_loss(
                digit_capsules=digit_capsules,
                targets=target,
                reconstructed=reconstructed,
                original_images=data
            )
            
            loss = loss_dict['total_loss']
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            total_margin_loss += loss_dict['margin_loss'].item()
            total_reconstruction_loss += loss_dict['reconstruction_loss'].item()
            
            # Accuracy
            predictions = self.model.predict(digit_capsules)
            correct += (predictions == target).sum().item()
            total += target.size(0)
            
        self.scheduler.step()
        
        return {
            'loss': total_loss / len(self.train_loader),
            'margin_loss': total_margin_loss / len(self.train_loader),
            'reconstruction_loss': total_reconstruction_loss / len(self.train_loader),
            'accuracy': correct / total
        }
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on validation set."""
        self.model.eval()
        total_loss = 0
        total_margin_loss = 0
        total_reconstruction_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                digit_capsules = outputs['digit_capsules']
                reconstructed = outputs.get('reconstructed')
                
                # Compute loss
                loss_dict = self.model.compute_loss(
                    digit_capsules=digit_capsules,
                    targets=target,
                    reconstructed=reconstructed,
                    original_images=data
                )
                
                # Statistics
                total_loss += loss_dict['total_loss'].item()
                total_margin_loss += loss_dict['margin_loss'].item()
                total_reconstruction_loss += loss_dict['reconstruction_loss'].item()
                
                # Accuracy
                predictions = self.model.predict(digit_capsules)
                correct += (predictions == target).sum().item()
                total += target.size(0)
        
        return {
            'loss': total_loss / len(self.val_loader),
            'margin_loss': total_margin_loss / len(self.val_loader),
            'reconstruction_loss': total_reconstruction_loss / len(self.val_loader),
            'accuracy': correct / total
        }
    
    def train_for_epochs(self, num_epochs: int = 50) -> Dict[str, List[float]]:
        """Train the model for specified number of epochs."""
        history = {
            'train_loss': [],
            'train_margin_loss': [],
            'train_reconstruction_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_margin_loss': [],
            'val_reconstruction_loss': [],
            'val_accuracy': []
        }
        
        for epoch in range(num_epochs):
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.evaluate()
            
            # Store results
            for key in train_metrics:
                history[f'train_{key}'].append(train_metrics[key])
                history[f'val_{key}'].append(val_metrics[key])
            
            # Print progress
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {train_metrics["loss"]:.4f}, Train Acc: {train_metrics["accuracy"]:.4f}')
            print(f'Val Loss: {val_metrics["loss"]:.4f}, Val Acc: {val_metrics["accuracy"]:.4f}')
        
        return history


if __name__ == "__main__":
    # Example usage
    batch_size = 4
    input_channels = 1
    image_size = 28  # MNIST
    
    # Create CapsNet
    capsnet = CapsNet(
        input_channels=input_channels,
        conv_channels=256,
        primary_capsules=32,
        primary_capsule_dim=8,
        digit_capsules=10,
        digit_capsule_dim=16,
        num_routing_iterations=3,
        use_reconstruction=True
    )
    
    # Create dummy input
    x = torch.randn(batch_size, input_channels, image_size, image_size)
    targets = torch.randint(0, 10, (batch_size,))
    
    # Forward pass
    outputs = capsnet(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Digit capsules shape: {outputs['digit_capsules'].shape}")
    print(f"Activities shape: {outputs['activities'].shape}")
    
    if outputs['reconstructed'] is not None:
        print(f"Reconstructed shape: {outputs['reconstructed'].shape}")
    
    # Compute loss
    loss_dict = capsnet.compute_loss(
        outputs['digit_capsules'],
        targets,
        outputs['reconstructed'],
        x
    )
    
    print(f"Total loss: {loss_dict['total_loss']:.4f}")
    print(f"Margin loss: {loss_dict['margin_loss']:.4f}")
    print(f"Reconstruction loss: {loss_dict['reconstruction_loss']:.4f}")
    
    # Make predictions
    predictions = capsnet.predict(outputs['digit_capsules'])
    print(f"Predictions: {predictions}")
    print(f"Targets: {targets}")
