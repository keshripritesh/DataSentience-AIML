"""
Advanced training pipeline for NeuralDicePredictor.

This module provides a comprehensive training system that combines
self-play, experience replay, and curriculum learning.
"""

import time
import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import torch
from tqdm import tqdm

from ..core.game_engine import GameEngine
from ..core.game_state import GameState, DiceAction
from .neural_net import NeuralAgent, NetworkConfig
from .mcts import AdvancedMCTS, MCTSConfig, MCTSPlayer


@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""
    num_episodes: int = 10000
    batch_size: int = 64
    learning_rate: float = 0.001
    experience_buffer_size: int = 10000
    self_play_games_per_update: int = 10
    evaluation_interval: int = 100
    save_interval: int = 500
    curriculum_learning: bool = True
    temperature_decay: float = 0.995
    min_temperature: float = 0.1


class ExperienceBuffer:
    """Experience replay buffer for training."""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def add_experience(self, state: np.ndarray, action: int, 
                      reward: float, next_state: np.ndarray, done: bool):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, 
                                                    np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch of experiences."""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self) -> int:
        return len(self.buffer)


class CurriculumLearning:
    """Curriculum learning system for progressive difficulty scaling."""
    
    def __init__(self, initial_difficulty: float = 0.1):
        self.current_difficulty = initial_difficulty
        self.difficulty_increase_rate = 0.01
        self.max_difficulty = 1.0
    
    def get_difficulty(self) -> float:
        """Get current difficulty level."""
        return self.current_difficulty
    
    def increase_difficulty(self):
        """Increase difficulty level."""
        self.current_difficulty = min(
            self.max_difficulty,
            self.current_difficulty + self.difficulty_increase_rate
        )
    
    def should_increase_difficulty(self, performance_metric: float) -> bool:
        """Check if difficulty should be increased based on performance."""
        return performance_metric > 0.7  # 70% success rate
    
    def get_game_parameters(self) -> Dict[str, Any]:
        """Get game parameters based on current difficulty."""
        # Adjust game complexity based on difficulty
        max_turns = int(5 + (15 - 5) * self.current_difficulty)
        max_rerolls = int(2 + (5 - 2) * self.current_difficulty)
        
        return {
            'max_turns': max_turns,
            'max_rerolls': max_rerolls,
            'scoring_complexity': self.current_difficulty
        }


class TrainingPipeline:
    """Advanced training pipeline for neural network agent."""
    
    def __init__(self, config: TrainingConfig, model_save_path: str = "models/"):
        self.config = config
        self.model_save_path = model_save_path
        
        # Initialize components
        self.game_engine = GameEngine()
        self.experience_buffer = ExperienceBuffer(config.experience_buffer_size)
        self.curriculum = CurriculumLearning()
        
        # Neural network configuration
        network_config = NetworkConfig(
            input_size=50,  # Adjust based on game state encoding
            hidden_sizes=(256, 256, 128),
            output_size=3,  # Number of actions
            learning_rate=config.learning_rate,
            dropout_rate=0.2,
            use_batch_norm=True
        )
        
        # Initialize neural agent
        self.neural_agent = NeuralAgent(network_config)
        
        # MCTS configuration
        mcts_config = MCTSConfig(
            simulation_count=100,
            exploration_constant=1.414,
            temperature=1.0,
            use_neural_network=True
        )
        
        # Initialize MCTS
        self.mcts = AdvancedMCTS(mcts_config, self.neural_agent)
        
        # Training statistics
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'losses': [],
            'win_rates': [],
            'curriculum_difficulty': []
        }
        
        # Performance tracking
        self.best_performance = 0.0
        self.episode_count = 0
    
    def train(self):
        """Main training loop."""
        print("Starting training pipeline...")
        print(f"Configuration: {self.config}")
        
        # Training loop
        for episode in tqdm(range(self.config.num_episodes), desc="Training Episodes"):
            self.episode_count = episode
            
            # Generate self-play games
            self._generate_self_play_games()
            
            # Update neural network
            if len(self.experience_buffer) >= self.config.batch_size:
                self._update_neural_network()
            
            # Evaluate performance
            if episode % self.config.evaluation_interval == 0:
                self._evaluate_performance()
            
            # Save model
            if episode % self.config.save_interval == 0:
                self._save_model(episode)
            
            # Update curriculum
            if self.config.curriculum_learning:
                self._update_curriculum()
            
            # Update temperature
            self._update_temperature()
        
        print("Training completed!")
        self._save_final_model()
    
    def _generate_self_play_games(self):
        """Generate self-play games for experience collection."""
        for _ in range(self.config.self_play_games_per_update):
            # Get game parameters from curriculum
            game_params = self.curriculum.get_game_parameters()
            
            # Create game state
            game_state = self.game_engine.create_initial_state(
                num_players=2,
                max_turns=game_params['max_turns']
            )
            
            # Play game
            game_history = self._play_game_with_mcts(game_state)
            
            # Store experiences
            self._store_game_experiences(game_history)
    
    def _play_game_with_mcts(self, game_state: GameState) -> List[Dict]:
        """Play a complete game using MCTS for decision making."""
        game_history = []
        current_state = game_state
        
        while not current_state.is_game_over:
            # Get current player state
            current_player = current_state.current_player_state
            
            # Use MCTS to select action
            action, action_data = self.mcts.search(current_state)
            
            # Execute action
            try:
                new_state = self.game_engine.execute_action(current_state, action, action_data)
                
                # Record experience
                experience = {
                    'state': current_state.to_tensor(),
                    'action': self._action_to_index(action),
                    'reward': self._calculate_reward(current_state, new_state),
                    'next_state': new_state.to_tensor(),
                    'done': new_state.is_game_over
                }
                game_history.append(experience)
                
                current_state = new_state
                
            except Exception as e:
                print(f"Action execution failed: {e}")
                break
        
        return game_history
    
    def _action_to_index(self, action: Tuple) -> int:
        """Convert action tuple to index."""
        action_type, _ = action
        if action_type == DiceAction.SCORE:
            return 0
        elif action_type == DiceAction.REROLL:
            return 1
        elif action_type == DiceAction.KEEP:
            return 2
        else:
            return 0
    
    def _calculate_reward(self, old_state: GameState, new_state: GameState) -> float:
        """Calculate reward for state transition."""
        if new_state.is_game_over:
            # Game ended
            winner = new_state.winner
            if winner == 0:  # Current player won
                return 1.0
            elif winner is not None:  # Current player lost
                return -1.0
            else:  # Tie game
                return 0.0
        
        # Score-based reward
        old_score = old_state.current_player_state.score
        new_score = new_state.current_player_state.score
        score_diff = new_score - old_score
        
        # Normalize score difference
        max_possible_score = 1000
        normalized_reward = score_diff / max_possible_score
        
        return normalized_reward
    
    def _store_game_experiences(self, game_history: List[Dict]):
        """Store game experiences in replay buffer."""
        for experience in game_history:
            self.experience_buffer.add_experience(
                experience['state'],
                experience['action'],
                experience['reward'],
                experience['next_state'],
                experience['done']
            )
    
    def _update_neural_network(self):
        """Update neural network using experience replay."""
        # Sample batch from experience buffer
        states, actions, rewards, next_states, dones = self.experience_buffer.sample_batch(
            self.config.batch_size
        )
        
        # Prepare target values
        target_values = self._calculate_target_values(rewards, next_states, dones)
        
        # Prepare target policies (one-hot encoded actions)
        target_policies = self._prepare_target_policies(actions)
        
        # Update network
        loss_info = self.neural_agent.update(states, target_policies, target_values)
        
        # Store loss information
        self.training_stats['losses'].append(loss_info)
    
    def _calculate_target_values(self, rewards: np.ndarray, next_states: np.ndarray, 
                               dones: np.ndarray) -> np.ndarray:
        """Calculate target values for training."""
        target_values = np.zeros_like(rewards)
        
        for i in range(len(rewards)):
            if dones[i]:
                target_values[i] = rewards[i]
            else:
                # Use neural network to estimate future value
                with torch.no_grad():
                    next_state_tensor = torch.FloatTensor(next_states[i:i+1]).to(
                        self.neural_agent.device
                    )
                    future_value = self.neural_agent.network.get_value(next_state_tensor).item()
                    target_values[i] = rewards[i] + 0.99 * future_value  # Discount factor
        
        return target_values
    
    def _prepare_target_policies(self, actions: np.ndarray) -> np.ndarray:
        """Prepare target policies as one-hot encoded vectors."""
        batch_size = len(actions)
        target_policies = np.zeros((batch_size, 3))  # 3 possible actions
        
        for i, action in enumerate(actions):
            target_policies[i, action] = 1.0
        
        return target_policies
    
    def _evaluate_performance(self):
        """Evaluate current performance."""
        # Play evaluation games
        win_rate = self._evaluate_win_rate()
        
        # Update statistics
        self.training_stats['win_rates'].append(win_rate)
        self.training_stats['curriculum_difficulty'].append(
            self.curriculum.get_difficulty()
        )
        
        # Update best performance
        if win_rate > self.best_performance:
            self.best_performance = win_rate
            print(f"New best performance: {win_rate:.3f}")
        
        print(f"Episode {self.episode_count}: Win Rate = {win_rate:.3f}, "
              f"Difficulty = {self.curriculum.get_difficulty():.3f}")
    
    def _evaluate_win_rate(self, num_games: int = 50) -> float:
        """Evaluate win rate against random opponent."""
        wins = 0
        
        for _ in range(num_games):
            # Create evaluation game
            game_state = self.game_engine.create_initial_state(num_players=2, max_turns=10)
            
            # Play game
            while not game_state.is_game_over:
                if game_state.current_player == 0:
                    # Neural agent's turn
                    action, action_data = self.mcts.search(game_state)
                else:
                    # Random opponent's turn
                    valid_actions = self.game_engine.get_valid_actions(game_state)
                    action = random.choice(valid_actions)
                    action_data = {} if action != DiceAction.KEEP else {'dice_indices': [0]}
                
                try:
                    game_state = self.game_engine.execute_action(game_state, action, action_data)
                except Exception:
                    break
            
            # Check winner
            if game_state.is_game_over and game_state.winner == 0:
                wins += 1
        
        return wins / num_games
    
    def _update_curriculum(self):
        """Update curriculum based on performance."""
        if len(self.training_stats['win_rates']) > 0:
            recent_performance = np.mean(self.training_stats['win_rates'][-10:])
            
            if self.curriculum.should_increase_difficulty(recent_performance):
                self.curriculum.increase_difficulty()
                print(f"Increased difficulty to {self.curriculum.get_difficulty():.3f}")
    
    def _update_temperature(self):
        """Update exploration temperature."""
        if hasattr(self.mcts, 'config'):
            self.mcts.config.temperature = max(
                self.config.min_temperature,
                self.mcts.config.temperature * self.config.temperature_decay
            )
    
    def _save_model(self, episode: int):
        """Save model checkpoint."""
        filename = f"model_episode_{episode}.pt"
        filepath = f"{self.model_save_path}/{filename}"
        
        self.neural_agent.save_model(filepath)
        print(f"Model saved: {filepath}")
    
    def _save_final_model(self):
        """Save final trained model."""
        filepath = f"{self.model_save_path}/final_model.pt"
        self.neural_agent.save_model(filepath)
        print(f"Final model saved: {filepath}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training results."""
        if not self.training_stats['win_rates']:
            return {}
        
        return {
            'final_win_rate': self.training_stats['win_rates'][-1],
            'best_win_rate': max(self.training_stats['win_rates']),
            'average_win_rate': np.mean(self.training_stats['win_rates']),
            'final_difficulty': self.curriculum.get_difficulty(),
            'total_episodes': self.episode_count,
            'final_loss': self.training_stats['losses'][-1] if self.training_stats['losses'] else None
        }
