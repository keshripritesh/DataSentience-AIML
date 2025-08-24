"""
Generative Adversarial Network with Style Transfer
This module implements a complete GAN system with style transfer capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from typing import Tuple, Optional, Dict, List
import math


class Generator(nn.Module):
    """
    Generator network for GAN with U-Net architecture.
    """
    
    def __init__(self, 
                 input_channels: int = 3,
                 output_channels: int = 3,
                 base_channels: int = 64,
                 num_blocks: int = 8):
        super(Generator, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.base_channels = base_channels
        
        # Encoder
        self.encoder = nn.ModuleList([
            nn.Conv2d(input_channels, base_channels, 4, 2, 1),  # 64
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),  # 128
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),  # 256
        ])
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(base_channels * 4) for _ in range(num_blocks)
        ])
        
        # Decoder
        self.decoder = nn.ModuleList([
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1),  # 128
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1),  # 64
            nn.ConvTranspose2d(base_channels, output_channels, 4, 2, 1),  # 3
        ])
        
        # Skip connections
        self.skip_connections = nn.ModuleList([
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, 1, 1),
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the generator.
        
        Args:
            x: Input image [batch_size, channels, height, width]
            
        Returns:
            Generated image [batch_size, channels, height, width]
        """
        # Store original size
        original_size = x.size()
        
        # Encoder
        encoder_features = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i < len(self.encoder) - 1:
                x = F.leaky_relu(x, 0.2)
                encoder_features.append(x)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Decoder with skip connections
        for i, layer in enumerate(self.decoder):
            x = layer(x)
            if i < len(self.decoder) - 1:
                # Add skip connection
                if i < len(encoder_features):
                    skip_feature = encoder_features[-(i+1)]
                    skip_feature = self.skip_connections[i](skip_feature)
                    x = x + skip_feature
                x = F.relu(x)
        
        # Ensure output has same size as input
        if x.size() != original_size:
            x = F.interpolate(x, size=original_size[2:], mode='bilinear', align_corners=False)
        
        return torch.tanh(x)


class ResidualBlock(nn.Module):
    """Residual block for the generator."""
    
    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return x + residual


class Discriminator(nn.Module):
    """
    PatchGAN discriminator.
    """
    
    def __init__(self, input_channels: int = 3, base_channels: int = 64):
        super(Discriminator, self).__init__()
        
        self.input_channels = input_channels
        self.base_channels = base_channels
        
        # PatchGAN architecture
        self.layers = nn.ModuleList([
            nn.Conv2d(input_channels, base_channels, 4, 2, 1),  # 64
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),  # 128
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),  # 256
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 1, 1),  # 512
            nn.Conv2d(base_channels * 8, 1, 4, 1, 1),  # 1
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the discriminator.
        
        Args:
            x: Input image [batch_size, channels, height, width]
            
        Returns:
            Patch predictions [batch_size, 1, height, width]
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.leaky_relu(x, 0.2)
        
        return x


class StyleTransferNetwork(nn.Module):
    """
    Neural Style Transfer network using VGG features.
    """
    
    def __init__(self):
        super(StyleTransferNetwork, self).__init__()
        
        # Load pre-trained VGG19
        vgg = models.vgg19(pretrained=True).features
        
        # Extract feature layers
        self.features = nn.ModuleList()
        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        
        layer_names = []
        for i, layer in enumerate(vgg):
            if isinstance(layer, nn.Conv2d):
                name = f'conv_{i}'
            elif isinstance(layer, nn.ReLU):
                name = f'relu_{i}'
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f'pool_{i}'
            elif isinstance(layer, nn.BatchNorm2d):
                name = f'bn_{i}'
            else:
                name = f'layer_{i}'
            
            self.features.append(layer)
            layer_names.append(name)
            
            if name in self.content_layers + self.style_layers:
                break
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract content and style features.
        
        Args:
            x: Input image [batch_size, channels, height, width]
            
        Returns:
            Dictionary containing content and style features
        """
        content_features = []
        style_features = []
        
        for i, layer in enumerate(self.features):
            x = layer(x)
            
            if f'conv_{i}' in self.content_layers:
                content_features.append(x)
            
            if f'conv_{i}' in self.style_layers:
                style_features.append(x)
        
        return {
            'content': content_features,
            'style': style_features
        }


class GANWithStyleTransfer(nn.Module):
    """
    Complete GAN system with style transfer capabilities.
    """
    
    def __init__(self,
                 input_channels: int = 3,
                 output_channels: int = 3,
                 base_channels: int = 64,
                 num_blocks: int = 8):
        super(GANWithStyleTransfer, self).__init__()
        
        self.generator = Generator(input_channels, output_channels, base_channels, num_blocks)
        self.discriminator = Discriminator(output_channels, base_channels)
        self.style_network = StyleTransferNetwork()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate stylized image."""
        return self.generator(x)
    
    def compute_content_loss(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute content loss between generated and target images."""
        gen_features = self.style_network(generated)
        target_features = self.style_network(target)
        
        content_loss = 0
        for gen_feat, target_feat in zip(gen_features['content'], target_features['content']):
            content_loss += F.mse_loss(gen_feat, target_feat)
        
        return content_loss
    
    def compute_style_loss(self, generated: torch.Tensor, style_image: torch.Tensor) -> torch.Tensor:
        """Compute style loss between generated and style images."""
        gen_features = self.style_network(generated)
        style_features = self.style_network(style_image)
        
        style_loss = 0
        for gen_feat, style_feat in zip(gen_features['style'], style_features['style']):
            # Compute Gram matrices
            gen_gram = self._gram_matrix(gen_feat)
            style_gram = self._gram_matrix(style_feat)
            style_loss += F.mse_loss(gen_gram, style_gram)
        
        return style_loss
    
    def _gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix for style loss."""
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)


class GANTrainer:
    """
    Trainer for GAN with style transfer.
    """
    
    def __init__(self,
                 model: GANWithStyleTransfer,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 lr_g: float = 0.0002,
                 lr_d: float = 0.0002):
        self.model = model.to(device)
        self.device = device
        
        # Optimizers
        self.optimizer_g = torch.optim.Adam(model.generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
        self.optimizer_d = torch.optim.Adam(model.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
        
        # Loss functions
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.content_loss = nn.MSELoss()
        
    def train_step(self, 
                  real_images: torch.Tensor,
                  style_images: torch.Tensor,
                  content_weight: float = 1.0,
                  style_weight: float = 10.0,
                  adversarial_weight: float = 1.0) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            real_images: Real images [batch_size, channels, height, width]
            style_images: Style images [batch_size, channels, height, width]
            content_weight: Weight for content loss
            style_weight: Weight for style loss
            adversarial_weight: Weight for adversarial loss
            
        Returns:
            Dictionary containing losses
        """
        batch_size = real_images.size(0)
        real_images = real_images.to(self.device)
        style_images = style_images.to(self.device)
        
        # Train Discriminator
        self.optimizer_d.zero_grad()
        
        # Generate fake images
        fake_images = self.model.generator(real_images)
        
        # Real images
        real_labels = torch.ones(batch_size, 1, 8, 8).to(self.device)  # PatchGAN output size
        fake_labels = torch.zeros(batch_size, 1, 8, 8).to(self.device)
        
        real_outputs = self.model.discriminator(real_images)
        fake_outputs = self.model.discriminator(fake_images.detach())
        
        d_loss_real = self.adversarial_loss(real_outputs, real_labels)
        d_loss_fake = self.adversarial_loss(fake_outputs, fake_labels)
        d_loss = (d_loss_real + d_loss_fake) / 2
        
        d_loss.backward()
        self.optimizer_d.step()
        
        # Train Generator
        self.optimizer_g.zero_grad()
        
        # Adversarial loss
        fake_outputs = self.model.discriminator(fake_images)
        g_loss_adv = self.adversarial_loss(fake_outputs, real_labels)
        
        # Content loss
        g_loss_content = self.model.compute_content_loss(fake_images, real_images)
        
        # Style loss
        g_loss_style = self.model.compute_style_loss(fake_images, style_images)
        
        # Total generator loss
        g_loss = (adversarial_weight * g_loss_adv + 
                 content_weight * g_loss_content + 
                 style_weight * g_loss_style)
        
        g_loss.backward()
        self.optimizer_g.step()
        
        return {
            'discriminator_loss': d_loss.item(),
            'generator_loss': g_loss.item(),
            'adversarial_loss': g_loss_adv.item(),
            'content_loss': g_loss_content.item(),
            'style_loss': g_loss_style.item()
        }


if __name__ == "__main__":
    # Example usage
    batch_size = 4
    channels = 3
    height = 256
    width = 256
    
    # Create GAN with style transfer
    gan = GANWithStyleTransfer(
        input_channels=channels,
        output_channels=channels,
        base_channels=64,
        num_blocks=8
    )
    
    # Create dummy data
    real_images = torch.randn(batch_size, channels, height, width)
    style_images = torch.randn(batch_size, channels, height, width)
    
    # Forward pass
    generated_images = gan(real_images)
    
    print(f"Real images shape: {real_images.shape}")
    print(f"Style images shape: {style_images.shape}")
    print(f"Generated images shape: {generated_images.shape}")
    
    # Compute losses
    content_loss = gan.compute_content_loss(generated_images, real_images)
    style_loss = gan.compute_style_loss(generated_images, style_images)
    
    print(f"Content loss: {content_loss:.4f}")
    print(f"Style loss: {style_loss:.4f}")
    
    # Test discriminator
    discriminator_output = gan.discriminator(generated_images)
    print(f"Discriminator output shape: {discriminator_output.shape}")
