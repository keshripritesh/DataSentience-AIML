"""
Advanced Transformer model for music generation in ImprovAI.
Features sophisticated attention mechanisms, positional encoding, and music-specific optimizations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, List, Dict
import numpy as np
import logging

from utils.config import get_config

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer model.
    Provides information about token positions in the sequence.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        self.max_seq_len = max_len

        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for Transformer.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, 
                                   V: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.
        
        Args:
            Q: Query tensor
            K: Key tensor
            V: Value tensor
            mask: Optional attention mask
            
        Returns:
            Tuple of (attention_output, attention_weights)
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through multi-head attention.
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            mask: Optional attention mask
            
        Returns:
            Tuple of (attention_output, attention_weights)
        """
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear transformation
        output = self.w_o(attention_output)
        
        return output, attention_weights


class TransformerBlock(nn.Module):
    """
    Transformer block with self-attention and feed-forward layers.
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super(TransformerBlock, self).__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor
            mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # Self-attention with residual connection
        attn_output, attn_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attn_weights


class MusicTransformer(nn.Module):
    """
    Advanced Transformer model for music generation.
    
    Features:
    - Multi-head self-attention
    - Positional encoding
    - Layer normalization
    - Dropout for regularization
    - Music-specific optimizations
    """
    
    def __init__(self,
                 vocab_size: int,
                 d_model: int = 512,
                 n_heads: int = 8,
                 n_layers: int = 6,
                 d_ff: int = 2048,
                 dropout: float = 0.1,
                 max_seq_len: int = 1024):
        super(MusicTransformer, self).__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout_rate = dropout
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"MusicTransformer initialized with vocab_size={vocab_size}, "
                   f"d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}")
    
    @property
    def device(self):
        """Get the device of the model."""
        return next(self.parameters()).device
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize embeddings
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        
        # Initialize transformer blocks
        for block in self.transformer_blocks:
            # Attention weights
            nn.init.xavier_uniform_(block.attention.w_q.weight)
            nn.init.xavier_uniform_(block.attention.w_k.weight)
            nn.init.xavier_uniform_(block.attention.w_v.weight)
            nn.init.xavier_uniform_(block.attention.w_o.weight)
            
            # Feed-forward weights
            nn.init.xavier_uniform_(block.feed_forward[0].weight)
            nn.init.xavier_uniform_(block.feed_forward[3].weight)
            
            # Initialize biases to zero
            nn.init.zeros_(block.attention.w_q.bias)
            nn.init.zeros_(block.attention.w_k.bias)
            nn.init.zeros_(block.attention.w_v.bias)
            nn.init.zeros_(block.attention.w_o.bias)
            nn.init.zeros_(block.feed_forward[0].bias)
            nn.init.zeros_(block.feed_forward[3].bias)
        
        # Initialize output projection
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
    
    def create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """
        Create causal mask for autoregressive generation.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Causal mask tensor
        """
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through the transformer model.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask
            
        Returns:
            Tuple of (logits, attention_weights_list)
        """
        batch_size, seq_len = input_ids.size()
        
        # Token embeddings
        embeddings = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        
        # Add positional encoding
        embeddings = embeddings.transpose(0, 1)  # (seq_len, batch_size, d_model)
        embeddings = self.pos_encoding(embeddings)
        embeddings = embeddings.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Apply dropout
        embeddings = self.dropout(embeddings)
        
        # Create causal mask for autoregressive generation
        causal_mask = self.create_causal_mask(seq_len).to(input_ids.device)
        
        # Process through transformer blocks
        hidden_states = embeddings
        attention_weights_list = []
        
        for block in self.transformer_blocks:
            hidden_states, attn_weights = block(hidden_states, causal_mask)
            attention_weights_list.append(attn_weights)
        
        # Project to vocabulary
        logits = self.output_projection(hidden_states)
        
        return logits, attention_weights_list
    
    def generate(self,
                 input_ids: torch.Tensor,
                 max_length: int = 64,
                 temperature: float = 1.0,
                 top_k: int = 50,
                 top_p: float = 0.9,
                 do_sample: bool = True,
                 pad_token_id: int = 0) -> torch.Tensor:
        """
        Generate music continuation.
        
        Args:
            input_ids: Input sequence of shape (batch_size, seq_len)
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling or greedy decoding
            pad_token_id: Padding token ID
            
        Returns:
            Generated sequence of shape (batch_size, max_length)
        """
        self.eval()
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Initialize output sequence
        generated = input_ids.clone()
        
        with torch.no_grad():
            # Generate only the remaining tokens
            remaining_length = max_length - input_ids.size(1)
            for _ in range(remaining_length):
                # Get model predictions
                logits, _ = self.forward(generated)
                
                # Get next token logits
                next_token_logits = logits[:, -1, :] / temperature
                
                if do_sample:
                    # Apply top-k filtering
                    if top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                        next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                        next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                    
                    # Apply nucleus sampling
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Sample from the filtered distribution
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated
    
    def get_loss(self, 
                 input_ids: torch.Tensor,
                 target_ids: torch.Tensor,
                 ignore_index: int = -100) -> torch.Tensor:
        """
        Calculate loss for training.
        
        Args:
            input_ids: Input sequence
            target_ids: Target sequence
            ignore_index: Index to ignore in loss calculation
            
        Returns:
            Loss tensor
        """
        logits, _ = self.forward(input_ids)
        
        # Reshape for loss calculation
        logits = logits.view(-1, self.vocab_size)
        targets = target_ids.view(-1)
        
        # Calculate cross-entropy loss
        loss = F.cross_entropy(logits, targets, ignore_index=ignore_index)
        
        return loss
    
    def save_model(self, filepath: str):
        """Save model to file."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'd_ff': self.d_ff,
            'dropout': self.dropout_rate,
            'max_seq_len': self.max_seq_len
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str, device: torch.device = None) -> "MusicTransformer":
        """Load model from file."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        
        model = cls(
            vocab_size=checkpoint['vocab_size'],
            d_model=checkpoint['d_model'],
            n_heads=checkpoint['n_heads'],
            n_layers=checkpoint['n_layers'],
            d_ff=checkpoint['d_ff'],
            dropout=checkpoint['dropout'],
            max_seq_len=checkpoint['max_seq_len']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        logger.info(f"Model loaded from {filepath}")
        return model


class MusicTransformerTrainer:
    """
    Trainer class for the MusicTransformer model.
    """
    
    def __init__(self, model: MusicTransformer, config=None):
        self.model = model
        self.config = config or get_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer with warmup
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.model.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = self._create_scheduler()
        
        logger.info(f"MusicTransformerTrainer initialized on device {self.device}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        # Warmup for first 10% of training steps
        warmup_steps = 1000
        
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * (step - warmup_steps) / 10000)))
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
        """
        Perform a single training step.
        
        Args:
            batch: Tuple of (input_ids, target_ids)
            
        Returns:
            Loss value
        """
        self.model.train()
        
        input_ids, target_ids = batch
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)
        
        # Forward pass
        loss = self.model.get_loss(input_ids, target_ids)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
    
    def validate(self, val_loader) -> float:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids, target_ids = batch
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                
                loss = self.model.get_loss(input_ids, target_ids)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return avg_loss
    
    def train(self, train_loader, val_loader=None, num_epochs: int = 10):
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            # Training
            train_losses = []
            for batch in train_loader:
                loss = self.train_step(batch)
                train_losses.append(loss)
            
            avg_train_loss = np.mean(train_losses)
            
            # Validation
            if val_loader is not None:
                avg_val_loss = self.validate(val_loader)
                logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                           f"Train Loss: {avg_train_loss:.4f}, "
                           f"Val Loss: {avg_val_loss:.4f}")
            else:
                logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                           f"Train Loss: {avg_train_loss:.4f}")
        
        logger.info("Training completed")
    
    def save_checkpoint(self, filepath: str, **kwargs):
        """Save training checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        checkpoint.update(kwargs)
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(f"Checkpoint loaded from {filepath}")
    
    def to(self, device: torch.device):
        """Move trainer to device."""
        self.device = device
        self.model = self.model.to(device)
        return self
