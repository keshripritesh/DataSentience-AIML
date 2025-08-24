"""
Capsule Layer Implementation
This module implements the core capsule layer with vector neurons and routing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import math


class CapsuleLayer(nn.Module):
    """
    A capsule layer that transforms input capsules to output capsules using routing.
    
    Each capsule is a vector that encodes both presence and pose information.
    """
    
    def __init__(self, 
                 num_in_capsules: int,
                 in_capsule_dim: int,
                 num_out_capsules: int,
                 out_capsule_dim: int,
                 num_routing_iterations: int = 3,
                 routing_method: str = 'dynamic'):
        super(CapsuleLayer, self).__init__()
        
        self.num_in_capsules = num_in_capsules
        self.in_capsule_dim = in_capsule_dim
        self.num_out_capsules = num_out_capsules
        self.out_capsule_dim = out_capsule_dim
        self.num_routing_iterations = num_routing_iterations
        self.routing_method = routing_method
        
        # Transformation matrices for each input-output capsule pair
        # Shape: (num_in_capsules, num_out_capsules, in_capsule_dim, out_capsule_dim)
        self.W = nn.Parameter(
            torch.randn(num_in_capsules, num_out_capsules, in_capsule_dim, out_capsule_dim) * 0.1
        )
        
        # Initialize routing weights (will be computed dynamically)
        self.routing_weights = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the capsule layer.
        
        Args:
            x: Input capsules [batch_size, num_in_capsules, in_capsule_dim]
            
        Returns:
            Output capsules [batch_size, num_out_capsules, out_capsule_dim]
        """
        batch_size = x.size(0)
        
        # Expand input for broadcasting
        # Shape: [batch_size, num_in_capsules, 1, in_capsule_dim]
        x_expanded = x.unsqueeze(2)
        
        # Transform input capsules using learned transformation matrices
        # Shape: [batch_size, num_in_capsules, num_out_capsules, out_capsule_dim]
        u_hat = torch.matmul(x_expanded, self.W).squeeze(2)
        
        # Apply routing algorithm
        if self.routing_method == 'dynamic':
            output_capsules = self._dynamic_routing(u_hat)
        elif self.routing_method == 'em':
            output_capsules = self._em_routing(u_hat)
        else:
            raise ValueError(f"Unknown routing method: {self.routing_method}")
        
        return output_capsules
    
    def _dynamic_routing(self, u_hat: torch.Tensor) -> torch.Tensor:
        """
        Dynamic routing by agreement algorithm.
        
        Args:
            u_hat: Transformed input capsules [batch_size, num_in_capsules, num_out_capsules, out_capsule_dim]
            
        Returns:
            Output capsules [batch_size, num_out_capsules, out_capsule_dim]
        """
        batch_size, num_in_capsules, num_out_capsules, out_capsule_dim = u_hat.size()
        
        # Initialize routing weights uniformly
        # Shape: [batch_size, num_in_capsules, num_out_capsules]
        b = torch.zeros(batch_size, num_in_capsules, num_out_capsules, device=u_hat.device)
        
        for iteration in range(self.num_routing_iterations):
            # Apply softmax to routing weights
            c = F.softmax(b, dim=2)
            
            # Weighted sum of input capsules
            # Shape: [batch_size, num_out_capsules, out_capsule_dim]
            s = torch.sum(c.unsqueeze(-1) * u_hat, dim=1)
            
            # Apply squashing function
            v = self._squash(s)
            
            # Update routing weights based on agreement
            if iteration < self.num_routing_iterations - 1:
                # Compute agreement between input and output capsules
                # Shape: [batch_size, num_in_capsules, num_out_capsules]
                agreement = torch.sum(u_hat * v.unsqueeze(1), dim=-1)
                b = b + agreement
        
        return v
    
    def _em_routing(self, u_hat: torch.Tensor) -> torch.Tensor:
        """
        Expectation-Maximization routing algorithm (simplified version).
        
        Args:
            u_hat: Transformed input capsules [batch_size, num_in_capsules, num_out_capsules, out_capsule_dim]
            
        Returns:
            Output capsules [batch_size, num_out_capsules, out_capsule_dim]
        """
        batch_size, num_in_capsules, num_out_capsules, out_capsule_dim = u_hat.size()
        
        # Initialize routing weights uniformly
        r = torch.ones(batch_size, num_in_capsules, num_out_capsules, device=u_hat.device) / num_out_capsules
        
        for iteration in range(self.num_routing_iterations):
            # M-step: Update output capsules
            r_expanded = r.unsqueeze(-1)
            s = torch.sum(r_expanded * u_hat, dim=1) / (torch.sum(r, dim=1, keepdim=True).unsqueeze(-1) + 1e-8)
            v = self._squash(s)
            
            # E-step: Update routing weights
            if iteration < self.num_routing_iterations - 1:
                # Compute similarity between input and output capsules
                similarity = torch.sum(u_hat * v.unsqueeze(1), dim=-1)
                r = F.softmax(similarity, dim=2)
        
        return v
    
    def _squash(self, s: torch.Tensor) -> torch.Tensor:
        """
        Squashing function that preserves vector direction while normalizing magnitude.
        
        Args:
            s: Input vectors [batch_size, num_capsules, capsule_dim]
            
        Returns:
            Squashed vectors [batch_size, num_capsules, capsule_dim]
        """
        # Compute squared norm
        squared_norm = torch.sum(s**2, dim=-1, keepdim=True)
        
        # Apply squashing function: v = ||s||^2 / (1 + ||s||^2) * s / ||s||
        scale = squared_norm / (1 + squared_norm)
        unit_vector = s / (torch.sqrt(squared_norm) + 1e-8)
        
        return scale * unit_vector


class PrimaryCapsuleLayer(nn.Module):
    """
    Primary capsule layer that converts CNN features to capsules.
    """
    
    def __init__(self, 
                 in_channels: int,
                 num_capsules: int,
                 capsule_dim: int,
                 kernel_size: int = 9,
                 stride: int = 2):
        super(PrimaryCapsuleLayer, self).__init__()
        
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        
        # Convolutional layer to create capsules
        self.conv = nn.Conv2d(
            in_channels, 
            num_capsules * capsule_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=0
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert CNN features to primary capsules.
        
        Args:
            x: CNN features [batch_size, in_channels, height, width]
            
        Returns:
            Primary capsules [batch_size, num_capsules, capsule_dim]
        """
        batch_size = x.size(0)
        
        # Apply convolution
        # Shape: [batch_size, num_capsules * capsule_dim, height, width]
        conv_out = self.conv(x)
        
        # Reshape to capsules
        # Shape: [batch_size, num_capsules, capsule_dim, height, width]
        capsules = conv_out.view(batch_size, self.num_capsules, self.capsule_dim, -1)
        
        # Transpose to get [batch_size, num_capsules, height*width, capsule_dim]
        capsules = capsules.transpose(2, 3)
        
        # Reshape to [batch_size, num_capsules * height * width, capsule_dim]
        capsules = capsules.contiguous().view(batch_size, -1, self.capsule_dim)
        
        # Apply squashing
        capsules = self._squash(capsules)
        
        return capsules
    
    def _squash(self, s: torch.Tensor) -> torch.Tensor:
        """Squashing function for primary capsules."""
        squared_norm = torch.sum(s**2, dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        unit_vector = s / (torch.sqrt(squared_norm) + 1e-8)
        return scale * unit_vector


class ReconstructionLayer(nn.Module):
    """
    Reconstruction layer that decodes capsules back to images.
    """
    
    def __init__(self, 
                 num_capsules: int,
                 capsule_dim: int,
                 input_dim: int,
                 hidden_dim: int = 512):
        super(ReconstructionLayer, self).__init__()
        
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        
        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(capsule_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, input_dim),
            nn.Sigmoid()  # Output in [0, 1] range
        )
        
    def forward(self, capsules: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct input from capsules.
        
        Args:
            capsules: Input capsules [batch_size, num_capsules, capsule_dim]
            target: Target images [batch_size, channels, height, width]
            
        Returns:
            Reconstructed images [batch_size, channels, height, width]
        """
        batch_size = capsules.size(0)
        
        # Flatten target for comparison
        target_flat = target.view(batch_size, -1)
        
        # Use the capsule with maximum activity for reconstruction
        # Compute capsule activities (magnitudes)
        activities = torch.norm(capsules, dim=-1)  # [batch_size, num_capsules]
        
        # Get the most active capsule for each sample
        max_activities, max_indices = torch.max(activities, dim=1)  # [batch_size]
        
        # Select the most active capsule for each sample
        # Shape: [batch_size, capsule_dim]
        selected_capsules = capsules[torch.arange(batch_size), max_indices]
        
        # Decode to image
        # Shape: [batch_size, input_dim]
        reconstructed_flat = self.decoder(selected_capsules)
        
        # Reshape back to image dimensions
        reconstructed = reconstructed_flat.view_as(target)
        
        return reconstructed


def compute_capsule_accuracy(capsules: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute accuracy based on capsule activities.
    
    Args:
        capsules: Output capsules [batch_size, num_classes, capsule_dim]
        targets: Target labels [batch_size]
        
    Returns:
        Accuracy
    """
    # Compute capsule activities (magnitudes)
    activities = torch.norm(capsules, dim=-1)  # [batch_size, num_classes]
    
    # Predict class with maximum activity
    predictions = torch.argmax(activities, dim=1)
    
    # Compute accuracy
    correct = (predictions == targets).float().sum()
    accuracy = correct / targets.size(0)
    
    return accuracy.item()


if __name__ == "__main__":
    # Example usage
    batch_size = 4
    num_in_capsules = 8
    in_capsule_dim = 16
    num_out_capsules = 10
    out_capsule_dim = 16
    
    # Create capsule layer
    capsule_layer = CapsuleLayer(
        num_in_capsules=num_in_capsules,
        in_capsule_dim=in_capsule_dim,
        num_out_capsules=num_out_capsules,
        out_capsule_dim=out_capsule_dim,
        num_routing_iterations=3
    )
    
    # Create input capsules
    x = torch.randn(batch_size, num_in_capsules, in_capsule_dim)
    
    # Forward pass
    output_capsules = capsule_layer(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output_capsules.shape}")
    print(f"Output capsule activities: {torch.norm(output_capsules, dim=-1)}")
