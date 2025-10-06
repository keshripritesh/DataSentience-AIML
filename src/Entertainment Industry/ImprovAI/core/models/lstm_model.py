"""
Advanced LSTM model for music generation in ImprovAI.
Features attention mechanisms, dropout, and sophisticated architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import numpy as np
import logging

from utils.config import get_config

logger = logging.getLogger(__name__)


class AttentionLayer(nn.Module):
    """
    Attention mechanism for LSTM model.
    Helps the model focus on relevant parts of the input sequence.
    """
    
    def __init__(self, hidden_size: int, attention_size: int = 128):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        
        # Attention weights
        self.attention_weights = nn.Linear(hidden_size, attention_size)
        self.attention_context = nn.Linear(attention_size, 1)
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention mechanism to hidden states.
        
        Args:
            hidden_states: Tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Tuple of (context_vector, attention_weights)
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # Calculate attention scores
        attention_scores = self.attention_weights(hidden_states)  # (batch_size, seq_len, attention_size)
        attention_scores = torch.tanh(attention_scores)
        attention_scores = self.attention_context(attention_scores)  # (batch_size, seq_len, 1)
        attention_scores = attention_scores.squeeze(-1)  # (batch_size, seq_len)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, seq_len)
        
        # Calculate context vector
        context_vector = torch.bmm(attention_weights.unsqueeze(1), hidden_states)  # (batch_size, 1, hidden_size)
        context_vector = context_vector.squeeze(1)  # (batch_size, hidden_size)
        
        return context_vector, attention_weights


class MusicLSTM(nn.Module):
    """
    Advanced LSTM model for music generation.
    
    Features:
    - Multi-layer LSTM with dropout
    - Attention mechanism
    - Residual connections
    - Batch normalization
    - Advanced output projection
    """
    
    def __init__(self, 
                 vocab_size: int,
                 embedding_dim: int = 256,
                 hidden_size: int = 512,
                 num_layers: int = 3,
                 dropout: float = 0.2,
                 use_attention: bool = True,
                 use_residual: bool = True):
        super(MusicLSTM, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_attention = use_attention
        self.use_residual = use_residual
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layers
        self.lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            input_size = embedding_dim if i == 0 else hidden_size
            lstm_layer = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=1,
                dropout=dropout if i < num_layers - 1 else 0.0,
                batch_first=True,
                bidirectional=False
            )
            self.lstm_layers.append(lstm_layer)
        
        # Attention layer
        if use_attention:
            self.attention = AttentionLayer(hidden_size)
        
        # Output projection layers
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, vocab_size)
        )
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
        # Layer normalization for each LSTM layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])
        
        # Residual projection layer for dimension matching
        if use_residual:
            self.residual_projection = nn.Linear(embedding_dim, hidden_size)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"MusicLSTM initialized with vocab_size={vocab_size}, "
                   f"embedding_dim={embedding_dim}, hidden_size={hidden_size}, "
                   f"num_layers={num_layers}")
    
    @property
    def device(self):
        """Get the device of the model."""
        return next(self.parameters()).device
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize embedding weights
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)
        
        # Initialize LSTM weights
        for lstm_layer in self.lstm_layers:
            for name, param in lstm_layer.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        
        # Initialize output projection weights
        for layer in self.output_projection:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, 
                input_ids: torch.Tensor,
                hidden_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the LSTM model.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            hidden_states: Optional initial hidden states
            
        Returns:
            Tuple of (logits, (final_hidden, final_cell))
        """
        batch_size, seq_len = input_ids.size()
        
        # Embed input tokens
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)
        
        # Process through LSTM layers
        lstm_output = embedded
        layer_hidden_states = []
        
        for i, lstm_layer in enumerate(self.lstm_layers):
            # Initialize hidden states for this layer if not provided
            if hidden_states is None:
                h0 = torch.zeros(1, batch_size, self.hidden_size, device=input_ids.device)
                c0 = torch.zeros(1, batch_size, self.hidden_size, device=input_ids.device)
                layer_hidden = (h0, c0)
            else:
                layer_hidden = hidden_states
            
            # Forward pass through LSTM layer
            lstm_output, layer_hidden = lstm_layer(lstm_output, layer_hidden)
            
            # Apply layer normalization
            lstm_output = self.layer_norms[i](lstm_output)
            
            # Store hidden states for attention
            layer_hidden_states.append(lstm_output)
            
            # Residual connection (skip first layer)
            if self.use_residual and i > 0:
                # Ensure tensors have the same size for residual connection
                if lstm_output.size(-1) != embedded.size(-1):
                    # Project embedded to match lstm_output size
                    embedded = self.residual_projection(embedded)
                lstm_output = lstm_output + embedded
        
        # Apply attention if enabled
        if self.use_attention:
            context_vector, attention_weights = self.attention(lstm_output)
            # Use context vector for final prediction
            final_hidden = context_vector
        else:
            # Use last hidden state
            final_hidden = lstm_output[:, -1, :]
            attention_weights = None
        
        # Apply batch normalization
        if batch_size > 1:
            final_hidden = self.batch_norm(final_hidden)
        
        # Project to vocabulary for all positions
        # We need to project the full sequence, not just the final hidden state
        if self.use_attention:
            # For attention, we use the context vector for each position
            logits = self.output_projection(lstm_output)  # (batch_size, seq_len, vocab_size)
        else:
            # For non-attention, we need to project each position
            # Reshape to (batch_size * seq_len, hidden_size)
            lstm_output_flat = lstm_output.view(-1, self.hidden_size)
            # Project to vocabulary
            logits_flat = self.output_projection(lstm_output_flat)  # (batch_size * seq_len, vocab_size)
            # Reshape back to (batch_size, seq_len, vocab_size)
            logits = logits_flat.view(batch_size, seq_len, self.vocab_size)
        
        return logits, (final_hidden, attention_weights)
    
    def generate(self,
                 input_ids: torch.Tensor,
                 max_length: int = 64,
                 temperature: float = 1.0,
                 top_k: int = 50,
                 top_p: float = 0.9,
                 do_sample: bool = True) -> torch.Tensor:
        """
        Generate music continuation.
        
        Args:
            input_ids: Input sequence of shape (batch_size, seq_len)
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling or greedy decoding
            
        Returns:
            Generated sequence of shape (batch_size, max_length)
        """
        self.eval()
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Initialize output sequence
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get model predictions
                logits, _ = self.forward(generated)
                
                # Get next token logits (handle both 2D and 3D logits)
                if logits.dim() == 3:
                    next_token_logits = logits[:, -1, :] / temperature
                else:
                    next_token_logits = logits / temperature
                
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
            'embedding_dim': self.embedding_dim,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'use_attention': self.use_attention,
            'use_residual': self.use_residual
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str, device: torch.device = None) -> "MusicLSTM":
        """Load model from file."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(filepath, map_location=device)
        
        model = cls(
            vocab_size=checkpoint['vocab_size'],
            embedding_dim=checkpoint['embedding_dim'],
            hidden_size=checkpoint['hidden_size'],
            num_layers=checkpoint['num_layers'],
            dropout=checkpoint['dropout'],
            use_attention=checkpoint['use_attention'],
            use_residual=checkpoint['use_residual']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        logger.info(f"Model loaded from {filepath}")
        return model


class MusicLSTMTrainer:
    """
    Trainer class for the MusicLSTM model.
    """
    
    def __init__(self, model: MusicLSTM, config=None):
        self.model = model
        self.config = config or get_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.model.learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6
        )
        
        logger.info(f"MusicLSTMTrainer initialized on device {self.device}")
    
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
            
            # Update learning rate
            self.scheduler.step()
        
        logger.info("Training completed")
    
    def save_checkpoint(self, filepath: str):
        """Save training checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(f"Checkpoint loaded from {filepath}")
