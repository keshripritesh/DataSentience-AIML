"""
Attention Mechanisms for FluidNetSim.

Implements various attention mechanisms for neural network architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import math
import logging

logger = logging.getLogger(__name__)

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True
    ):
        """
        Initialize multi-head attention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.scaling = self.head_dim ** -0.5
        
        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query tensor
            key: Key tensor (if None, uses query)
            value: Value tensor (if None, uses key)
            attn_mask: Attention mask
            
        Returns:
            Output tensor and attention weights
        """
        if key is None:
            key = query
        if value is None:
            value = key
        
        batch_size, seq_len, embed_dim = query.size()
        
        # Linear projections and reshape
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        # Apply attention mask if provided
        if attn_mask is not None:
            attn_weights = attn_weights.masked_fill(attn_mask == 0, float('-inf'))
        
        # Apply softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attn_weights

class SpatialAttention(nn.Module):
    """Spatial attention mechanism for 2D feature maps."""
    
    def __init__(
        self,
        channels: int,
        reduction: int = 8,
        use_spatial: bool = True
    ):
        """
        Initialize spatial attention.
        
        Args:
            channels: Number of input channels
            reduction: Reduction factor for channel attention
            use_spatial: Whether to use spatial attention
        """
        super().__init__()
        
        self.channels = channels
        self.reduction = reduction
        self.use_spatial = use_spatial
        
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.channel_fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        # Spatial attention
        if use_spatial:
            self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial attention mechanism."""
        batch_size, channels, height, width = x.size()
        
        # Channel attention
        avg_out = self.channel_fc(self.avg_pool(x))
        max_out = self.channel_fc(self.max_pool(x))
        channel_out = self.sigmoid(avg_out + max_out)
        
        # Apply channel attention
        x = x * channel_out
        
        # Spatial attention
        if self.use_spatial:
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            spatial_out = torch.cat([avg_out, max_out], dim=1)
            spatial_out = self.sigmoid(self.spatial_conv(spatial_out))
            x = x * spatial_out
        
        return x

class TemporalAttention(nn.Module):
    """Temporal attention mechanism for sequence modeling."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize temporal attention.
        
        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        
        self.scaling = self.head_dim ** -0.5
        
        # Linear projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of temporal attention.
        
        Args:
            x: Input tensor (batch, seq_len, hidden_dim)
            mask: Attention mask
            
        Returns:
            Output tensor and attention weights
        """
        batch_size, seq_len, hidden_dim = x.size()
        
        # Linear projections and reshape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        # Apply mask if provided
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, hidden_dim
        )
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attn_weights

class CrossAttention(nn.Module):
    """Cross-attention mechanism for multi-modal fusion."""
    
    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        value_dim: int,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize cross-attention.
        
        Args:
            query_dim: Query dimension
            key_dim: Key dimension
            value_dim: Value dimension
            embed_dim: Output embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.scaling = self.head_dim ** -0.5
        
        # Linear projections
        self.q_proj = nn.Linear(query_dim, embed_dim)
        self.k_proj = nn.Linear(key_dim, embed_dim)
        self.v_proj = nn.Linear(value_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of cross-attention.
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            attn_mask: Attention mask
            
        Returns:
            Output tensor and attention weights
        """
        batch_size, seq_len, embed_dim = query.size()
        
        # Linear projections and reshape
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        # Apply mask if provided
        if attn_mask is not None:
            attn_weights = attn_weights.masked_fill(attn_mask == 0, float('-inf'))
        
        # Apply softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attn_weights

class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward layers."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        """
        Initialize transformer block.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            dropout: Dropout probability
            activation: Activation function
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        
        # Self-attention
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of transformer block.
        
        Args:
            x: Input tensor
            attn_mask: Attention mask
            
        Returns:
            Output tensor
        """
        # Self-attention with residual connection
        attn_output, _ = self.self_attn(x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        
        return x

class AttentionModule(nn.Module):
    """Main attention module that combines different attention mechanisms."""
    
    def __init__(
        self,
        channels: int,
        attention_type: str = "spatial",
        **kwargs
    ):
        """
        Initialize attention module.
        
        Args:
            channels: Number of channels
            attention_type: Type of attention mechanism
            **kwargs: Additional parameters
        """
        super().__init__()
        
        self.channels = channels
        self.attention_type = attention_type
        
        if attention_type == "spatial":
            self.attention = SpatialAttention(channels, **kwargs)
        elif attention_type == "temporal":
            self.attention = TemporalAttention(channels, **kwargs)
        elif attention_type == "multihead":
            self.attention = MultiHeadAttention(channels, **kwargs)
        elif attention_type == "transformer":
            self.attention = TransformerBlock(channels, **kwargs)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass of attention module."""
        return self.attention(x, **kwargs)
    
    def __repr__(self) -> str:
        return f"AttentionModule(type={self.attention_type}, channels={self.channels})"
