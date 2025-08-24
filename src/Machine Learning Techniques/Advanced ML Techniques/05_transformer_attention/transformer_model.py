"""
Transformer Architecture with Attention
This module implements a complete transformer with self-attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer sequences.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input embeddings [seq_len, batch_size, d_model]
            
        Returns:
            Embeddings with positional encoding
        """
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through multi-head attention.
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            mask: Attention mask [batch_size, seq_len, seq_len]
            
        Returns:
            Output and attention weights
        """
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1)  # Add head dimension
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)
        
        return output, attention_weights


class FeedForward(nn.Module):
    """
    Feed-forward network for transformer blocks.
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(FeedForward, self).__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feed-forward network."""
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single transformer block with self-attention and feed-forward.
    """
    
    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 d_ff: int, 
                 dropout: float = 0.1):
        super(TransformerBlock, self).__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Attention mask [batch_size, seq_len, seq_len]
            
        Returns:
            Output and attention weights
        """
        # Self-attention with residual connection
        attn_output, attention_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attention_weights


class TransformerEncoder(nn.Module):
    """
    Transformer encoder stack.
    """
    
    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 num_layers: int, 
                 d_ff: int, 
                 dropout: float = 0.1):
        super(TransformerEncoder, self).__init__()
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, 
                x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through encoder stack.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Attention mask [batch_size, seq_len, seq_len]
            
        Returns:
            Output and attention weights from all layers
        """
        attention_weights = []
        
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attention_weights.append(attn_weights)
        
        return x, attention_weights


class TransformerDecoder(nn.Module):
    """
    Transformer decoder stack.
    """
    
    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 num_layers: int, 
                 d_ff: int, 
                 dropout: float = 0.1):
        super(TransformerDecoder, self).__init__()
        
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, 
                x: torch.Tensor, 
                encoder_output: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through decoder stack.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            encoder_output: Output from encoder
            src_mask: Source attention mask
            tgt_mask: Target attention mask
            
        Returns:
            Output and attention weights from all layers
        """
        attention_weights = []
        
        for layer in self.layers:
            x, attn_weights = layer(x, encoder_output, src_mask, tgt_mask)
            attention_weights.append(attn_weights)
        
        return x, attention_weights


class TransformerDecoderBlock(nn.Module):
    """
    Single transformer decoder block.
    """
    
    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 d_ff: int, 
                 dropout: float = 0.1):
        super(TransformerDecoderBlock, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                x: torch.Tensor, 
                encoder_output: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through decoder block.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            encoder_output: Output from encoder
            src_mask: Source attention mask
            tgt_mask: Target attention mask
            
        Returns:
            Output and attention weights
        """
        # Self-attention
        attn_output, self_attn_weights = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention
        attn_output, cross_attn_weights = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        attention_weights = {
            'self_attention': self_attn_weights,
            'cross_attention': cross_attn_weights
        }
        
        return x, attention_weights


class Transformer(nn.Module):
    """
    Complete transformer model with encoder and decoder.
    """
    
    def __init__(self,
                 vocab_size: int,
                 d_model: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 d_ff: int = 2048,
                 max_len: int = 5000,
                 dropout: float = 0.1):
        super(Transformer, self).__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Encoder and decoder
        self.encoder = TransformerEncoder(d_model, num_heads, num_layers, d_ff, dropout)
        self.decoder = TransformerDecoder(d_model, num_heads, num_layers, d_ff, dropout)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                src: torch.Tensor, 
                tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]:
        """
        Forward pass through transformer.
        
        Args:
            src: Source sequence [batch_size, src_len]
            tgt: Target sequence [batch_size, tgt_len]
            src_mask: Source attention mask
            tgt_mask: Target attention mask
            
        Returns:
            Output logits and attention weights
        """
        # Embeddings and positional encoding
        src_embedded = self.dropout(self.pos_encoding(self.embedding(src).transpose(0, 1)).transpose(0, 1))
        tgt_embedded = self.dropout(self.pos_encoding(self.embedding(tgt).transpose(0, 1)).transpose(0, 1))
        
        # Encoder
        encoder_output, encoder_attention = self.encoder(src_embedded, src_mask)
        
        # Decoder
        decoder_output, decoder_attention = self.decoder(tgt_embedded, encoder_output, src_mask, tgt_mask)
        
        # Output projection
        output = self.output_projection(decoder_output)
        
        attention_weights = {
            'encoder': encoder_attention,
            'decoder': decoder_attention
        }
        
        return output, attention_weights
    
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for decoder."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class TransformerLM(nn.Module):
    """
    Transformer language model (GPT-style).
    """
    
    def __init__(self,
                 vocab_size: int,
                 d_model: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 d_ff: int = 2048,
                 max_len: int = 5000,
                 dropout: float = 0.1):
        super(TransformerLM, self).__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through language model.
        
        Args:
            x: Input sequence [batch_size, seq_len]
            mask: Attention mask [batch_size, seq_len, seq_len]
            
        Returns:
            Output logits and attention weights
        """
        # Embeddings and positional encoding
        x = self.dropout(self.pos_encoding(self.embedding(x).transpose(0, 1)).transpose(0, 1))
        
        # Transformer blocks
        attention_weights = []
        for block in self.transformer_blocks:
            x, attn_weights = block(x, mask)
            attention_weights.append(attn_weights)
        
        # Output projection
        output = self.output_projection(x)
        
        return output, attention_weights


if __name__ == "__main__":
    # Example usage
    batch_size = 4
    seq_len = 20
    vocab_size = 1000
    d_model = 512
    
    # Create transformer
    transformer = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=8,
        num_layers=6
    )
    
    # Create dummy data
    src = torch.randint(0, vocab_size, (batch_size, seq_len))
    tgt = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Create masks
    src_mask = None
    tgt_mask = transformer.generate_square_subsequent_mask(seq_len)
    
    # Forward pass
    output, attention_weights = transformer(src, tgt, src_mask, tgt_mask)
    
    print(f"Source shape: {src.shape}")
    print(f"Target shape: {tgt.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of encoder attention layers: {len(attention_weights['encoder'])}")
    print(f"Number of decoder attention layers: {len(attention_weights['decoder'])}")
    
    # Test language model
    lm = TransformerLM(vocab_size=vocab_size, d_model=d_model)
    lm_output, lm_attention = lm(src, tgt_mask)
    
    print(f"Language model output shape: {lm_output.shape}")
    print(f"Language model attention layers: {len(lm_attention)}")
