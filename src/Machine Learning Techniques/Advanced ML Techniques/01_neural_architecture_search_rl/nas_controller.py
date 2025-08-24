"""
Neural Architecture Search Controller using Reinforcement Learning
This module implements an LSTM-based controller that generates neural network architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple
import random


class NASController(nn.Module):
    """
    LSTM-based controller for generating neural network architectures.
    
    The controller generates sequences of actions that define:
    - Layer types (conv, pool, fc, etc.)
    - Layer parameters (filters, kernel size, etc.)
    - Connections between layers
    """
    
    def __init__(self, 
                 vocab_size: int,
                 hidden_size: int = 100,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super(NASController, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM controller
        self.lstm = nn.LSTM(
            input_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output projection
        self.decoder = nn.Linear(hidden_size, vocab_size)
        
        # Temperature for exploration
        self.temperature = 1.0
        
    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor] = None):
        """
        Forward pass through the controller.
        
        Args:
            x: Input sequence [batch_size, seq_len, vocab_size]
            hidden: Initial hidden state (h0, c0)
            
        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
            hidden: Final hidden state (hn, cn)
        """
        lstm_out, hidden = self.lstm(x, hidden)
        logits = self.decoder(lstm_out)
        return logits, hidden
    
    def sample_architecture(self, 
                          max_layers: int = 10,
                          temperature: float = None) -> List[int]:
        """
        Sample a complete neural network architecture.
        
        Args:
            max_layers: Maximum number of layers to generate
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            List of action indices representing the architecture
        """
        if temperature is None:
            temperature = self.temperature
            
        self.eval()
        with torch.no_grad():
            # Start with start token
            current_input = torch.zeros(1, 1, self.vocab_size)
            current_input[0, 0, 0] = 1.0  # START token
            
            hidden = None
            actions = []
            
            for _ in range(max_layers):
                # Get logits from controller
                logits, hidden = self.forward(current_input, hidden)
                
                # Sample next action
                probs = F.softmax(logits[0, -1] / temperature, dim=0)
                action = torch.multinomial(probs, 1).item()
                
                # Stop if we hit end token
                if action == 1:  # END token
                    break
                    
                actions.append(action)
                
                # Prepare next input
                next_input = torch.zeros(1, 1, self.vocab_size)
                next_input[0, 0, action] = 1.0
                current_input = next_input
                
        return actions
    
    def get_action_log_probs(self, 
                            actions: List[int],
                            temperature: float = None) -> torch.Tensor:
        """
        Get log probabilities of a sequence of actions.
        
        Args:
            actions: List of action indices
            temperature: Temperature for probability calculation
            
        Returns:
            Log probabilities of the actions
        """
        if temperature is None:
            temperature = self.temperature
            
        self.eval()
        with torch.no_grad():
            # Convert actions to one-hot
            inputs = torch.zeros(1, len(actions), self.vocab_size)
            for i, action in enumerate(actions):
                inputs[0, i, action] = 1.0
                
            # Get logits
            logits, _ = self.forward(inputs)
            
            # Calculate log probabilities
            log_probs = F.log_softmax(logits / temperature, dim=-1)
            
            # Extract log probs for the taken actions
            action_log_probs = []
            for i, action in enumerate(actions):
                action_log_probs.append(log_probs[0, i, action])
                
        return torch.stack(action_log_probs)


class PolicyGradientTrainer:
    """
    Trainer for the NAS controller using policy gradient.
    """
    
    def __init__(self, controller: NASController, lr: float = 0.001):
        self.controller = controller
        self.optimizer = torch.optim.Adam(controller.parameters(), lr=lr)
        
    def update_policy(self, 
                     actions_list: List[List[int]], 
                     rewards: List[float],
                     baseline: float = None):
        """
        Update the controller policy using policy gradient.
        
        Args:
            actions_list: List of action sequences
            rewards: List of corresponding rewards
            baseline: Baseline reward for variance reduction
        """
        self.controller.train()
        
        # Calculate baseline if not provided
        if baseline is None:
            baseline = np.mean(rewards)
            
        # Calculate advantages
        advantages = [r - baseline for r in rewards]
        
        # Update policy
        total_loss = 0
        for actions, advantage in zip(actions_list, advantages):
            # Get log probabilities
            log_probs = self.controller.get_action_log_probs(actions)
            
            # Calculate loss (negative because we want to maximize reward)
            loss = -torch.sum(log_probs) * advantage
            total_loss += loss
            
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()


# Architecture vocabulary
ARCHITECTURE_VOCAB = {
    0: 'START',
    1: 'END',
    2: 'CONV_3x3',
    3: 'CONV_5x5', 
    4: 'CONV_7x7',
    5: 'MAX_POOL_3x3',
    6: 'AVG_POOL_3x3',
    7: 'SEPARABLE_CONV_3x3',
    8: 'SEPARABLE_CONV_5x5',
    9: 'DILATED_CONV_3x3',
    10: 'DILATED_CONV_5x5',
    11: 'FULLY_CONNECTED',
    12: 'BATCH_NORM',
    13: 'DROPOUT',
    14: 'RELU',
    15: 'SIGMOID',
    16: 'TANH',
    17: 'SKIP_CONNECTION',
    18: 'CONCATENATE'
}


def decode_architecture(actions: List[int]) -> List[str]:
    """Decode action indices to architecture description."""
    return [ARCHITECTURE_VOCAB.get(action, f'UNKNOWN_{action}') for action in actions]


if __name__ == "__main__":
    # Example usage
    vocab_size = len(ARCHITECTURE_VOCAB)
    controller = NASController(vocab_size=vocab_size)
    trainer = PolicyGradientTrainer(controller)
    
    # Sample an architecture
    architecture = controller.sample_architecture(max_layers=8)
    print("Generated Architecture:")
    print(decode_architecture(architecture))
