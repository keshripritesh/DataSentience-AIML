"""
Neural Radiance Fields (NeRF) Implementation
This module implements NeRF for novel view synthesis and 3D scene representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding for high-frequency features.
    """
    
    def __init__(self, input_dim: int, max_freq_log2: int = 10, num_freqs: int = None):
        super(PositionalEncoding, self).__init__()
        
        self.input_dim = input_dim
        
        if num_freqs is None:
            num_freqs = max_freq_log2 * 2
        
        # Create frequency bands
        freq_bands = 2.0 ** torch.linspace(0.0, max_freq_log2, steps=num_freqs)
        
        # Create encoding matrix
        self.register_buffer('freq_bands', freq_bands)
        
        # Output dimension: input_dim * (2 * num_freqs + 1)
        self.output_dim = input_dim * (2 * num_freqs + 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding.
        
        Args:
            x: Input tensor [..., input_dim]
            
        Returns:
            Encoded tensor [..., output_dim]
        """
        # Expand dimensions for broadcasting
        x_expanded = x.unsqueeze(-2)  # [..., 1, input_dim]
        freq_bands = self.freq_bands.view(1, -1, 1)  # [1, num_freqs, 1]
        
        # Compute sin and cos
        sin_terms = torch.sin(freq_bands * x_expanded)  # [..., num_freqs, input_dim]
        cos_terms = torch.cos(freq_bands * x_expanded)  # [..., num_freqs, input_dim]
        
        # Concatenate all terms
        encoded = torch.cat([
            x.unsqueeze(-2),  # Original coordinates
            sin_terms,
            cos_terms
        ], dim=-2)  # [..., 2*num_freqs+1, input_dim]
        
        # Flatten
        encoded = encoded.flatten(-2)  # [..., output_dim]
        
        return encoded


class NeRFMLP(nn.Module):
    """
    MLP network for NeRF.
    """
    
    def __init__(self, 
                 pos_encoding_dim: int,
                 view_encoding_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 8,
                 skip_connection: int = 4):
        super(NeRFMLP, self).__init__()
        
        self.pos_encoding_dim = pos_encoding_dim
        self.view_encoding_dim = view_encoding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.skip_connection = skip_connection
        
        # Position encoding layers (first part)
        self.pos_layers = nn.ModuleList()
        
        # Input layer
        self.pos_layers.append(nn.Linear(pos_encoding_dim, hidden_dim))
        
        # Hidden layers
        for i in range(1, num_layers):
            if i == skip_connection:
                # Skip connection: concatenate input
                self.pos_layers.append(nn.Linear(hidden_dim + pos_encoding_dim, hidden_dim))
            else:
                self.pos_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # View-dependent layers (second part)
        self.view_layers = nn.ModuleList()
        
        # Concatenate view direction
        self.view_layers.append(nn.Linear(hidden_dim + view_encoding_dim, hidden_dim // 2))
        self.view_layers.append(nn.Linear(hidden_dim // 2, 3))  # RGB output
        
        # Density output (from position layers)
        self.density_layer = nn.Linear(hidden_dim, 1)
        
    def forward(self, pos_encoded: torch.Tensor, view_encoded: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through NeRF MLP.
        
        Args:
            pos_encoded: Position encoding [..., pos_encoding_dim]
            view_encoded: View direction encoding [..., view_encoding_dim]
            
        Returns:
            Tuple of (density, rgb)
        """
        x = pos_encoded
        
        # Position layers
        for i, layer in enumerate(self.pos_layers):
            if i == self.skip_connection:
                # Skip connection
                x = torch.cat([x, pos_encoded], dim=-1)
            x = layer(x)
            x = F.relu(x)
        
        # Density output
        density = self.density_layer(x)
        
        # View-dependent layers
        view_input = torch.cat([x, view_encoded], dim=-1)
        
        for i, layer in enumerate(self.view_layers):
            view_input = layer(view_input)
            if i < len(self.view_layers) - 1:
                view_input = F.relu(view_input)
        
        rgb = torch.sigmoid(view_input)  # Output in [0, 1]
        
        return density, rgb


class VolumeRendering:
    """
    Volume rendering utilities.
    """
    
    @staticmethod
    def integrate_along_ray(density: torch.Tensor, 
                           rgb: torch.Tensor, 
                           z_vals: torch.Tensor,
                           rays_d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Integrate along ray using volume rendering equation.
        
        Args:
            density: Volume density [num_rays, num_samples]
            rgb: RGB values [num_rays, num_samples, 3]
            z_vals: Sample distances [num_rays, num_samples]
            rays_d: Ray directions [num_rays, 3]
            
        Returns:
            Tuple of (rgb_final, depth, weights)
        """
        # Compute distances between samples
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.tensor([1e10], device=dists.device).expand(dists[..., :1].shape)], dim=-1)
        
        # Compute alpha values
        alpha = 1.0 - torch.exp(-density.squeeze(-1) * dists)
        
        # Compute transmittance
        transmittance = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
        transmittance = torch.cat([torch.ones_like(transmittance[..., :1]), transmittance[..., :-1]], dim=-1)
        
        # Compute weights
        weights = alpha * transmittance
        
        # Compute final RGB
        rgb_final = torch.sum(weights.unsqueeze(-1) * rgb, dim=-2)
        
        # Compute depth
        depth = torch.sum(weights * z_vals, dim=-1)
        
        return rgb_final, depth, weights
    
    @staticmethod
    def sample_rays(rays_o: torch.Tensor, 
                   rays_d: torch.Tensor,
                   near: float, 
                   far: float, 
                   num_samples: int,
                   perturb: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample points along rays.
        
        Args:
            rays_o: Ray origins [num_rays, 3]
            rays_d: Ray directions [num_rays, 3]
            near: Near plane distance
            far: Far plane distance
            num_samples: Number of samples per ray
            perturb: Whether to add random perturbation
            
        Returns:
            Tuple of (points, z_vals)
        """
        num_rays = rays_o.shape[0]
        
        # Sample distances
        t_vals = torch.linspace(0., 1., steps=num_samples, device=rays_o.device)
        z_vals = near * (1. - t_vals) + far * t_vals
        
        # Add perturbation
        if perturb:
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
            lower = torch.cat([z_vals[..., :1], mids], dim=-1)
            t_rand = torch.rand(z_vals.shape, device=rays_o.device)
            z_vals = lower + (upper - lower) * t_rand
        
        # Expand for all rays
        z_vals = z_vals.expand(num_rays, num_samples)
        
        # Compute 3D points
        points = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(-1)
        
        return points, z_vals


class HierarchicalNeRF(nn.Module):
    """
    Hierarchical NeRF with coarse and fine networks.
    """
    
    def __init__(self, 
                 pos_encoding_dim: int,
                 view_encoding_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 8,
                 num_samples_coarse: int = 64,
                 num_samples_fine: int = 128):
        super(HierarchicalNeRF, self).__init__()
        
        self.pos_encoding_dim = pos_encoding_dim
        self.view_encoding_dim = view_encoding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_samples_coarse = num_samples_coarse
        self.num_samples_fine = num_samples_fine
        
        # Coarse network
        self.coarse_net = NeRFMLP(pos_encoding_dim, view_encoding_dim, hidden_dim, num_layers)
        
        # Fine network
        self.fine_net = NeRFMLP(pos_encoding_dim, view_encoding_dim, hidden_dim, num_layers)
        
        # Positional encodings
        self.pos_encoding = PositionalEncoding(3, max_freq_log2=10)
        self.view_encoding = PositionalEncoding(3, max_freq_log2=4)
        
    def forward(self, 
                rays_o: torch.Tensor, 
                rays_d: torch.Tensor,
                near: float = 0.0,
                far: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hierarchical NeRF.
        
        Args:
            rays_o: Ray origins [num_rays, 3]
            rays_d: Ray directions [num_rays, 3]
            near: Near plane distance
            far: Far plane distance
            
        Returns:
            Dictionary containing coarse and fine outputs
        """
        num_rays = rays_o.shape[0]
        
        # Coarse sampling
        points_coarse, z_vals_coarse = VolumeRendering.sample_rays(
            rays_o, rays_d, near, far, self.num_samples_coarse, perturb=True
        )
        
        # Encode coarse points and directions
        pos_encoded_coarse = self.pos_encoding(points_coarse)
        view_encoded_coarse = self.view_encoding(rays_d.unsqueeze(1).expand(-1, self.num_samples_coarse, -1))
        
        # Coarse network forward pass
        density_coarse, rgb_coarse = self.coarse_net(pos_encoded_coarse, view_encoded_coarse)
        
        # Coarse volume rendering
        rgb_coarse_final, depth_coarse, weights_coarse = VolumeRendering.integrate_along_ray(
            density_coarse, rgb_coarse, z_vals_coarse, rays_d
        )
        
        # Fine sampling using importance sampling
        z_vals_fine = self._importance_sampling(z_vals_coarse, weights_coarse, near, far)
        
        # Encode fine points and directions
        points_fine = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals_fine.unsqueeze(-1)
        pos_encoded_fine = self.pos_encoding(points_fine)
        view_encoded_fine = self.view_encoding(rays_d.unsqueeze(1).expand(-1, self.num_samples_fine, -1))
        
        # Fine network forward pass
        density_fine, rgb_fine = self.fine_net(pos_encoded_fine, view_encoded_fine)
        
        # Fine volume rendering
        rgb_fine_final, depth_fine, weights_fine = VolumeRendering.integrate_along_ray(
            density_fine, rgb_fine, z_vals_fine, rays_d
        )
        
        return {
            'rgb_coarse': rgb_coarse_final,
            'rgb_fine': rgb_fine_final,
            'depth_coarse': depth_coarse,
            'depth_fine': depth_fine,
            'weights_coarse': weights_coarse,
            'weights_fine': weights_fine
        }
    
    def _importance_sampling(self, 
                           z_vals_coarse: torch.Tensor, 
                           weights_coarse: torch.Tensor,
                           near: float, 
                           far: float) -> torch.Tensor:
        """
        Importance sampling for fine network.
        
        Args:
            z_vals_coarse: Coarse sample distances [num_rays, num_samples_coarse]
            weights_coarse: Coarse weights [num_rays, num_samples_coarse]
            near: Near plane distance
            far: Far plane distance
            
        Returns:
            Fine sample distances [num_rays, num_samples_fine]
        """
        num_rays = z_vals_coarse.shape[0]
        
        # Add noise to weights for better sampling
        weights_coarse = weights_coarse + 1e-5
        
        # Normalize weights
        weights_coarse = weights_coarse / torch.sum(weights_coarse, dim=-1, keepdim=True)
        
        # Sample from coarse weights
        z_vals_mid = 0.5 * (z_vals_coarse[..., 1:] + z_vals_coarse[..., :-1])
        
        # Sample fine points
        fine_samples = torch.multinomial(weights_coarse, self.num_samples_fine, replacement=True)
        fine_samples = fine_samples.sort(dim=-1)[0]
        
        # Get fine z_vals
        z_vals_fine = torch.gather(z_vals_mid, -1, fine_samples)
        
        # Add uniform samples
        z_vals_uniform = torch.linspace(near, far, self.num_samples_fine, device=z_vals_coarse.device)
        z_vals_uniform = z_vals_uniform.expand(num_rays, self.num_samples_fine)
        
        # Combine and sort
        z_vals_fine = torch.cat([z_vals_fine, z_vals_uniform], dim=-1)
        z_vals_fine, _ = torch.sort(z_vals_fine, dim=-1)
        
        return z_vals_fine


class NeRFTrainer:
    """
    Trainer for NeRF models.
    """
    
    def __init__(self, 
                 model: HierarchicalNeRF,
                 lr: float = 5e-4,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
    def train_step(self, 
                  rays_o: torch.Tensor, 
                  rays_d: torch.Tensor,
                  target_rgb: torch.Tensor) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            rays_o: Ray origins [num_rays, 3]
            rays_d: Ray directions [num_rays, 3]
            target_rgb: Target RGB values [num_rays, 3]
            
        Returns:
            Dictionary containing losses
        """
        # Forward pass
        outputs = self.model(rays_o, rays_d)
        
        # Compute losses
        loss_coarse = self.criterion(outputs['rgb_coarse'], target_rgb)
        loss_fine = self.criterion(outputs['rgb_fine'], target_rgb)
        
        # Total loss
        total_loss = loss_coarse + loss_fine
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'coarse_loss': loss_coarse.item(),
            'fine_loss': loss_fine.item()
        }


if __name__ == "__main__":
    # Example usage
    num_rays = 1024
    pos_encoding_dim = 63  # 3 * (2 * 10 + 1)
    view_encoding_dim = 27  # 3 * (2 * 4 + 1)
    
    # Create dummy data
    rays_o = torch.randn(num_rays, 3)
    rays_d = torch.randn(num_rays, 3)
    rays_d = F.normalize(rays_d, dim=-1)  # Normalize directions
    target_rgb = torch.rand(num_rays, 3)
    
    # Create NeRF model
    nerf = HierarchicalNeRF(
        pos_encoding_dim=pos_encoding_dim,
        view_encoding_dim=view_encoding_dim,
        hidden_dim=256,
        num_layers=8,
        num_samples_coarse=64,
        num_samples_fine=128
    )
    
    # Forward pass
    outputs = nerf(rays_o, rays_d)
    
    print(f"Rays shape: {rays_o.shape}")
    print(f"Coarse RGB shape: {outputs['rgb_coarse'].shape}")
    print(f"Fine RGB shape: {outputs['rgb_fine'].shape}")
    print(f"Coarse depth shape: {outputs['depth_coarse'].shape}")
    print(f"Fine depth shape: {outputs['depth_fine'].shape}")
    
    # Test trainer
    trainer = NeRFTrainer(nerf)
    losses = trainer.train_step(rays_o, rays_d, target_rgb)
    print(f"Training losses: {losses}")
    
    # Test positional encoding
    pos_encoding = PositionalEncoding(3)
    points = torch.randn(10, 3)
    encoded = pos_encoding(points)
    print(f"Positional encoding: {points.shape} -> {encoded.shape}")
