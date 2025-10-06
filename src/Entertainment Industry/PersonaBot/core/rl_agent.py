"""
Reinforcement Learning Agent for PersonaBot
Handles personality adaptation through RL algorithms
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import random
import logging
from dataclasses import dataclass

from config.settings import settings
from core.reward import RewardFunction, ConversationState
from core.personality import PersonalityEncoder

logger = logging.getLogger(__name__)

@dataclass
class Experience:
    """Represents a single experience for RL training"""
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    timestamp: float

class ActorCriticNetwork(nn.Module):
    """Actor-Critic neural network for personality adaptation"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(ActorCriticNetwork, self).__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor (policy) network
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()  # Output in [-1, 1] range
        )
        
        # Critic (value) network
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through both actor and critic"""
        shared_features = self.shared(state)
        action = self.actor(shared_features)
        value = self.critic(shared_features)
        return action, value

class RLAgent:
    """Advanced Reinforcement Learning agent for personality adaptation"""
    
    def __init__(self, personality_encoder: PersonalityEncoder):
        """Initialize RL agent"""
        self.personality_encoder = personality_encoder
        self.config = settings.rl
        
        # State and action dimensions
        self.state_dim = len(settings.personality.traits) + 10  # personality + context
        self.action_dim = len(settings.personality.traits)
        
        # Initialize neural networks
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actor_critic = ActorCriticNetwork(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.config.learning_rate)
        
        # Experience replay buffer
        self.memory = deque(maxlen=self.config.memory_size)
        
        # Training parameters
        self.epsilon = self.config.epsilon
        self.gamma = self.config.gamma
        
        # Performance tracking
        self.training_stats = {
            'episodes': 0,
            'total_reward': 0.0,
            'average_reward': 0.0,
            'loss_history': [],
            'epsilon_history': []
        }
        
        # Reward function
        self.reward_function = RewardFunction()
        
        logger.info(f"RL Agent initialized on {self.device}")
    
    def get_state(self, conversation_state: ConversationState) -> np.ndarray:
        """Convert conversation state to RL state vector"""
        # Personality vector
        personality_vector = conversation_state.personality_vector
        
        # Context features
        context_features = self._extract_context_features(conversation_state)
        
        # Combine into state vector
        state = np.concatenate([personality_vector, context_features])
        
        return state
    
    def _extract_context_features(self, conversation_state: ConversationState) -> np.ndarray:
        """Extract context features from conversation state"""
        features = []
        
        # Conversation length (normalized)
        features.append(min(conversation_state.conversation_length / 20.0, 1.0))
        
        # Time since start (normalized)
        time_since_start = (conversation_state.last_response_time - conversation_state.start_time).total_seconds()
        features.append(min(time_since_start / 300.0, 1.0))  # 5 minutes max
        
        # Engagement metrics
        engagement_metrics = conversation_state.engagement_metrics
        features.extend([
            engagement_metrics.get('sentiment', 0.0),
            engagement_metrics.get('engagement', 0.0),
            engagement_metrics.get('relevance', 0.0),
            engagement_metrics.get('coherence', 0.0)
        ])
        
        # Message diversity (simple heuristic)
        if conversation_state.messages:
            unique_words = len(set(' '.join([msg.get('content', '') for msg in conversation_state.messages]).split()))
            total_words = len(' '.join([msg.get('content', '') for msg in conversation_state.messages]).split())
            diversity = unique_words / max(total_words, 1)
            features.append(diversity)
        else:
            features.append(0.0)
        
        # Response time (normalized)
        response_time = (conversation_state.last_response_time - conversation_state.start_time).total_seconds()
        features.append(min(response_time / 60.0, 1.0))  # 1 minute max
        
        # Fill remaining slots with zeros
        while len(features) < 10:
            features.append(0.0)
        
        return np.array(features[:10])
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            # Random action
            action = np.random.uniform(-1.0, 1.0, self.action_dim)
        else:
            # Policy action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action, _ = self.actor_critic(state_tensor)
            action = action.cpu().numpy().flatten()
        
        return action
    
    def adapt_personality(self, 
                         action: np.ndarray,
                         learning_rate: float = 0.1) -> np.ndarray:
        """Apply action to adapt personality"""
        current_personality = self.personality_encoder.get_personality_vector()
        
        # Apply action as personality adaptation
        new_personality = current_personality + action * learning_rate
        
        # Clamp to valid range [0, 1]
        new_personality = np.clip(new_personality, 0.0, 1.0)
        
        # Convert action to feedback format for personality encoder
        feedback = {}
        for i, trait in enumerate(settings.personality.traits):
            feedback[trait] = action[i]
        
        # Update personality
        self.personality_encoder.adapt_personality(feedback, learning_rate)
        
        return new_personality
    
    def store_experience(self, 
                        state: np.ndarray,
                        action: np.ndarray,
                        reward: float,
                        next_state: np.ndarray,
                        done: bool) -> None:
        """Store experience in replay buffer"""
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            timestamp=random.random()  # Simple timestamp
        )
        
        self.memory.append(experience)
    
    def train(self, batch_size: Optional[int] = None) -> Dict[str, float]:
        """Train the agent on a batch of experiences"""
        if batch_size is None:
            batch_size = self.config.batch_size
        
        if len(self.memory) < batch_size:
            return {'actor_loss': 0.0, 'critic_loss': 0.0, 'total_loss': 0.0}
        
        # Sample batch
        batch = random.sample(self.memory, batch_size)
        
        # Prepare tensors
        states = torch.FloatTensor([exp.state for exp in batch]).to(self.device)
        actions = torch.FloatTensor([exp.action for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp.reward for exp in batch]).to(self.device)
        next_states = torch.FloatTensor([exp.next_state for exp in batch]).to(self.device)
        dones = torch.BoolTensor([exp.done for exp in batch]).to(self.device)
        
        # Forward pass
        current_actions, current_values = self.actor_critic(states)
        _, next_values = self.actor_critic(next_states)
        
        # Calculate target values
        target_values = rewards + (self.gamma * next_values.squeeze() * ~dones)
        target_values = target_values.detach()
        
        # Calculate losses
        critic_loss = nn.MSELoss()(current_values.squeeze(), target_values)
        actor_loss = -torch.mean(current_values.squeeze())  # Simple policy gradient
        
        total_loss = actor_loss + critic_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 1.0)
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon * self.config.epsilon_decay, self.config.epsilon_min)
        
        # Update training stats
        self.training_stats['loss_history'].append(total_loss.item())
        self.training_stats['epsilon_history'].append(self.epsilon)
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def process_interaction(self, 
                          user_message: str,
                          bot_response: str,
                          conversation_state: ConversationState) -> Tuple[float, Dict[str, float]]:
        """Process a single interaction and update the agent"""
        # Get current state
        current_state = self.get_state(conversation_state)
        
        # Calculate reward
        reward, detailed_rewards = self.reward_function.calculate_reward(
            user_message, bot_response, conversation_state, conversation_state.personality_vector
        )
        
        # Select action
        action = self.select_action(current_state, training=True)
        
        # Apply action to adapt personality
        new_personality = self.adapt_personality(action)
        
        # Update conversation state
        conversation_state.personality_vector = new_personality
        conversation_state.last_response_time = conversation_state.start_time  # Update time
        
        # Get next state
        next_state = self.get_state(conversation_state)
        
        # Determine if episode is done
        done = conversation_state.conversation_length >= self.config.update_frequency
        
        # Store experience
        self.store_experience(current_state, action, reward, next_state, done)
        
        # Train if enough experiences
        if len(self.memory) >= self.config.batch_size:
            training_losses = self.train()
            detailed_rewards.update(training_losses)
        
        # Update training stats
        self.training_stats['total_reward'] += reward
        self.training_stats['episodes'] += 1
        self.training_stats['average_reward'] = (
            self.training_stats['total_reward'] / self.training_stats['episodes']
        )
        
        return reward, detailed_rewards
    
    def get_policy_info(self) -> Dict[str, Any]:
        """Get information about the current policy"""
        return {
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'training_stats': self.training_stats.copy(),
            'device': str(self.device),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model"""
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
            'epsilon': self.epsilon,
            'config': self.config.__dict__
        }, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint['training_stats']
        self.epsilon = checkpoint['epsilon']
        
        logger.info(f"Model loaded from {filepath}")
    
    def reset(self) -> None:
        """Reset the agent's state"""
        self.memory.clear()
        self.training_stats = {
            'episodes': 0,
            'total_reward': 0.0,
            'average_reward': 0.0,
            'loss_history': [],
            'epsilon_history': []
        }
        self.epsilon = self.config.epsilon
        
        logger.info("RL Agent reset")
    
    def get_action_explanation(self, action: np.ndarray) -> Dict[str, str]:
        """Explain what the action means for personality adaptation"""
        explanation = {}
        
        for i, trait in enumerate(settings.personality.traits):
            action_value = action[i]
            current_value = self.personality_encoder.get_personality_vector()[i]
            
            if action_value > 0.1:
                explanation[trait] = f"Increase {trait} (current: {current_value:.2f}, change: +{action_value:.2f})"
            elif action_value < -0.1:
                explanation[trait] = f"Decrease {trait} (current: {current_value:.2f}, change: {action_value:.2f})"
            else:
                explanation[trait] = f"Maintain {trait} (current: {current_value:.2f}, change: {action_value:.2f})"
        
        return explanation
