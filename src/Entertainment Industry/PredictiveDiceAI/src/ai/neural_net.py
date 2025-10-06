"""
Advanced neural network architecture for NeuralDicePredictor.

This module provides sophisticated neural network models for
policy evaluation and value estimation in the dice game.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class NetworkConfig:
    """Configuration for neural network architecture."""
    input_size: int
    hidden_sizes: Tuple[int, ...] = (256, 256, 128)
    output_size: int = 3  # Policy head size
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    use_batch_norm: bool = True
    activation: str = "relu"


class AttentionModule(nn.Module):
    """Multi-head attention mechanism for pattern recognition."""
    
    def __init__(self, input_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        assert input_dim % num_heads == 0, "Input dimension must be divisible by num_heads"
        
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.projection = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Multi-head attention
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # Output projection and residual connection
        output = self.projection(context)
        output = self.layer_norm(output + x)
        
        return output


class ResidualBlock(nn.Module):
    """Residual block with batch normalization and dropout."""
    
    def __init__(self, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        # First layer
        x = self.linear1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second layer
        x = self.linear2(x)
        x = self.batch_norm2(x)
        
        # Residual connection
        x = F.relu(x + residual)
        x = self.dropout(x)
        
        return x


class DiceGameNeuralNet(nn.Module):
    """Advanced neural network for dice game AI."""
    
    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        
        # Input processing layers
        self.input_norm = nn.LayerNorm(config.input_size)
        self.input_dropout = nn.Dropout(config.dropout_rate)
        
        # Hidden layers with residual connections
        self.hidden_layers = nn.ModuleList()
        prev_size = config.input_size
        
        for hidden_size in config.hidden_sizes:
            if config.use_batch_norm:
                self.hidden_layers.append(ResidualBlock(prev_size, config.dropout_rate))
            else:
                self.hidden_layers.append(nn.Sequential(
                    nn.Linear(prev_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(config.dropout_rate)
                ))
            prev_size = hidden_size
        
        # Attention mechanism for complex pattern recognition
        if len(config.hidden_sizes) > 0:
            self.attention = AttentionModule(
                config.hidden_sizes[-1], 
                num_heads=4, 
                dropout=config.dropout_rate
            )
        
        # Policy head (action probabilities)
        self.policy_head = nn.Sequential(
            nn.Linear(prev_size, prev_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(prev_size // 2, config.output_size)
        )
        
        # Value head (state evaluation)
        self.value_head = nn.Sequential(
            nn.Linear(prev_size, prev_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(prev_size // 2, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Tuple of (policy_logits, value)
        """
        # Input normalization and dropout
        x = self.input_norm(x)
        x = self.input_dropout(x)
        
        # Hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
        
        # Attention mechanism (reshape for attention)
        if hasattr(self, 'attention'):
            batch_size = x.size(0)
            x = x.unsqueeze(1)  # Add sequence dimension
            x = self.attention(x)
            x = x.squeeze(1)  # Remove sequence dimension
        
        # Policy and value heads
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        
        return policy_logits, value
    
    def get_policy(self, x: torch.Tensor) -> torch.Tensor:
        """Get action probabilities from input state."""
        policy_logits, _ = self.forward(x)
        return F.softmax(policy_logits, dim=-1)
    
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Get state value from input state."""
        _, value = self.forward(x)
        return value.squeeze(-1)


class NeuralAgent:
    """Neural network-based AI agent for dice game."""
    
    def __init__(self, config: NetworkConfig, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)
        
        # Initialize neural network
        self.network = DiceGameNeuralNet(config).to(self.device)
        
        # Optimizer and loss function
        self.optimizer = optim.AdamW(
            self.network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=1000, T_mult=2
        )
        
        # Training statistics
        self.training_stats = {
            'total_loss': 0.0,
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'num_updates': 0
        }
    
    def select_action(self, state: np.ndarray, temperature: float = 1.0) -> int:
        """
        Select action based on current state.
        
        Args:
            state: Game state as numpy array
            temperature: Temperature for action selection (higher = more random)
            
        Returns:
            Selected action index
        """
        self.network.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            policy_logits, _ = self.network(state_tensor)
            
            if temperature == 0:
                # Greedy selection
                action = torch.argmax(policy_logits).item()
            else:
                # Temperature-scaled selection
                scaled_logits = policy_logits / temperature
                policy = F.softmax(scaled_logits, dim=0)
                action = torch.multinomial(policy, 1).item()
            
            return action
    
    def get_action_probabilities(self, state: np.ndarray) -> np.ndarray:
        """Get action probabilities for given state."""
        self.network.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            policy, _ = self.network(state_tensor)
            return F.softmax(policy, dim=-1).cpu().numpy().flatten()
    
    def update(self, states: np.ndarray, target_policies: np.ndarray, 
               target_values: np.ndarray) -> Dict[str, float]:
        """
        Update network parameters using supervised learning.
        
        Args:
            states: Batch of game states
            target_policies: Target action probabilities
            target_values: Target state values
            
        Returns:
            Dictionary with loss information
        """
        self.network.train()
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        target_policies_tensor = torch.FloatTensor(target_policies).to(self.device)
        target_values_tensor = torch.FloatTensor(target_values).to(self.device)
        
        # Forward pass
        policy_logits, values = self.network(states_tensor)
        
        # Calculate losses
        policy_loss = F.cross_entropy(policy_logits, target_policies_tensor)
        value_loss = F.mse_loss(values.squeeze(-1), target_values_tensor)
        
        # Total loss
        total_loss = policy_loss + value_loss
        
        # Backward pass and optimization
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Update statistics
        self.training_stats['total_loss'] += total_loss.item()
        self.training_stats['policy_loss'] += policy_loss.item()
        self.training_stats['value_loss'] += value_loss.item()
        self.training_stats['num_updates'] += 1
        
        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        }
    
    def save_model(self, filepath: str):
        """Save model to file."""
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'training_stats': self.training_stats
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load model from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
    
    def get_training_stats(self) -> Dict[str, float]:
        """Get average training statistics."""
        if self.training_stats['num_updates'] == 0:
            return {'total_loss': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0}
        
        return {
            'avg_total_loss': self.training_stats['total_loss'] / self.training_stats['num_updates'],
            'avg_policy_loss': self.training_stats['policy_loss'] / self.training_stats['num_updates'],
            'avg_value_loss': self.training_stats['value_loss'] / self.training_stats['num_updates'],
            'num_updates': self.training_stats['num_updates']
        }
    
    def reset_stats(self):
        """Reset training statistics."""
        self.training_stats = {
            'total_loss': 0.0,
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'num_updates': 0
        }
