"""
Main Training Loop for Neural Architecture Search with Reinforcement Learning
This module orchestrates the entire NAS training process.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import json
from typing import List, Dict, Tuple, Optional
from collections import deque
import matplotlib.pyplot as plt

from nas_controller import NASController, PolicyGradientTrainer, decode_architecture
from child_network import ChildNetwork, ChildNetworkTrainer


class NASTrainingLoop:
    """
    Main training loop for Neural Architecture Search.
    """
    
    def __init__(self,
                 controller: NASController,
                 train_loader,
                 val_loader,
                 num_episodes: int = 100,
                 child_epochs: int = 10,
                 max_layers: int = 10,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.controller = controller
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_episodes = num_episodes
        self.child_epochs = child_epochs
        self.max_layers = max_layers
        self.device = device
        
        # Training components
        self.policy_trainer = PolicyGradientTrainer(controller)
        
        # History tracking
        self.episode_rewards = []
        self.episode_accuracies = []
        self.episode_architectures = []
        self.best_reward = -float('inf')
        self.best_architecture = None
        
        # Experience replay for stability
        self.experience_buffer = deque(maxlen=100)
        
    def compute_reward(self, accuracy: float, num_params: int, training_time: float) -> float:
        """
        Compute reward based on accuracy, model complexity, and training efficiency.
        
        Args:
            accuracy: Validation accuracy
            num_params: Number of parameters in the model
            training_time: Time taken to train the model
            
        Returns:
            Reward value
        """
        # Base reward from accuracy
        reward = accuracy
        
        # Penalty for model complexity (encourage smaller models)
        param_penalty = min(num_params / 1e6, 0.1)  # Cap penalty at 10%
        reward -= param_penalty
        
        # Penalty for training time (encourage faster training)
        time_penalty = min(training_time / 60, 0.05)  # Cap penalty at 5%
        reward -= time_penalty
        
        return reward
    
    def train_child_network(self, architecture_actions: List[int]) -> Tuple[float, int, float]:
        """
        Train a child network with the given architecture.
        
        Args:
            architecture_actions: List of actions defining the architecture
            
        Returns:
            Tuple of (accuracy, num_parameters, training_time)
        """
        try:
            # Build child network
            input_shape = (3, 32, 32)  # CIFAR-10 default
            num_classes = 10
            
            child_net = ChildNetwork(input_shape, num_classes, architecture_actions)
            trainer = ChildNetworkTrainer(child_net, self.train_loader, self.val_loader, self.device)
            
            # Train the network
            start_time = time.time()
            history = trainer.train_for_epochs(self.child_epochs)
            training_time = time.time() - start_time
            
            # Get final validation accuracy
            final_accuracy = history['val_acc'][-1]
            num_parameters = child_net.get_num_parameters()
            
            return final_accuracy, num_parameters, training_time
            
        except Exception as e:
            print(f"Error training child network: {e}")
            return 0.0, 0, 0.0
    
    def run_episode(self, episode: int) -> Dict:
        """
        Run a single episode of architecture search.
        
        Args:
            episode: Episode number
            
        Returns:
            Episode results dictionary
        """
        print(f"\n=== Episode {episode + 1}/{self.num_episodes} ===")
        
        # Sample architecture from controller
        architecture_actions = self.controller.sample_architecture(max_layers=self.max_layers)
        architecture_str = decode_architecture(architecture_actions)
        
        print(f"Generated Architecture: {architecture_str}")
        
        # Train child network
        print("Training child network...")
        accuracy, num_params, training_time = self.train_child_network(architecture_actions)
        
        # Compute reward
        reward = self.compute_reward(accuracy, num_params, training_time)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Parameters: {num_params:,}")
        print(f"Training Time: {training_time:.2f}s")
        print(f"Reward: {reward:.4f}")
        
        # Store experience
        self.experience_buffer.append({
            'actions': architecture_actions,
            'reward': reward,
            'accuracy': accuracy,
            'num_params': num_params,
            'training_time': training_time
        })
        
        # Update best if improved
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_architecture = architecture_actions
            print(f"🎉 New best architecture found! Reward: {reward:.4f}")
        
        return {
            'episode': episode,
            'actions': architecture_actions,
            'architecture': architecture_str,
            'accuracy': accuracy,
            'num_params': num_params,
            'training_time': training_time,
            'reward': reward
        }
    
    def update_controller(self, batch_size: int = 10):
        """
        Update the controller using policy gradient.
        
        Args:
            batch_size: Number of experiences to use for update
        """
        if len(self.experience_buffer) < batch_size:
            return
        
        # Sample batch from experience buffer
        batch = list(self.experience_buffer)[-batch_size:]
        
        actions_list = [exp['actions'] for exp in batch]
        rewards = [exp['reward'] for exp in batch]
        
        # Update controller policy
        loss = self.policy_trainer.update_policy(actions_list, rewards)
        
        print(f"Controller updated. Policy loss: {loss:.4f}")
    
    def run_training(self) -> Dict:
        """
        Run the complete NAS training process.
        
        Returns:
            Training results and history
        """
        print("🚀 Starting Neural Architecture Search Training")
        print(f"Device: {self.device}")
        print(f"Episodes: {self.num_episodes}")
        print(f"Child epochs per episode: {self.child_epochs}")
        
        start_time = time.time()
        
        for episode in range(self.num_episodes):
            # Run episode
            episode_result = self.run_episode(episode)
            
            # Store results
            self.episode_rewards.append(episode_result['reward'])
            self.episode_accuracies.append(episode_result['accuracy'])
            self.episode_architectures.append(episode_result['architecture'])
            
            # Update controller periodically
            if (episode + 1) % 5 == 0:
                self.update_controller()
            
            # Print progress
            avg_reward = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else np.mean(self.episode_rewards)
            print(f"Average reward (last 10): {avg_reward:.4f}")
        
        total_time = time.time() - start_time
        
        print(f"\n🎯 Training Complete!")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Best reward: {self.best_reward:.4f}")
        print(f"Best architecture: {decode_architecture(self.best_architecture)}")
        
        return {
            'episode_rewards': self.episode_rewards,
            'episode_accuracies': self.episode_accuracies,
            'episode_architectures': self.episode_architectures,
            'best_reward': self.best_reward,
            'best_architecture': self.best_architecture,
            'total_time': total_time
        }
    
    def plot_training_progress(self, results: Dict):
        """
        Plot training progress.
        
        Args:
            results: Training results dictionary
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot rewards
        ax1.plot(results['episode_rewards'])
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True)
        
        # Plot accuracies
        ax2.plot(results['episode_accuracies'])
        ax2.set_title('Episode Accuracies')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)
        
        # Plot moving average rewards
        window = 10
        moving_avg = np.convolve(results['episode_rewards'], 
                                np.ones(window)/window, mode='valid')
        ax3.plot(range(window-1, len(results['episode_rewards'])), moving_avg)
        ax3.set_title(f'Moving Average Reward (window={window})')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Average Reward')
        ax3.grid(True)
        
        # Plot reward distribution
        ax4.hist(results['episode_rewards'], bins=20, alpha=0.7)
        ax4.set_title('Reward Distribution')
        ax4.set_xlabel('Reward')
        ax4.set_ylabel('Frequency')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('nas_training_progress.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, results: Dict, filename: str = 'nas_results.json'):
        """
        Save training results to file.
        
        Args:
            results: Training results dictionary
            filename: Output filename
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {
            'episode_rewards': [float(r) for r in results['episode_rewards']],
            'episode_accuracies': [float(a) for a in results['episode_accuracies']],
            'episode_architectures': results['episode_architectures'],
            'best_reward': float(results['best_reward']),
            'best_architecture': results['best_architecture'],
            'total_time': float(results['total_time'])
        }
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {filename}")


def create_dummy_data_loaders():
    """
    Create dummy data loaders for demonstration.
    In practice, you would use real datasets like CIFAR-10.
    """
    # Create dummy data
    batch_size = 32
    num_samples = 1000
    
    # Dummy training data
    train_data = torch.randn(num_samples, 3, 32, 32)
    train_labels = torch.randint(0, 10, (num_samples,))
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Dummy validation data
    val_data = torch.randn(num_samples//4, 3, 32, 32)
    val_labels = torch.randint(0, 10, (num_samples//4,))
    val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Example usage
    print("🧠 Neural Architecture Search with Reinforcement Learning")
    
    # Create dummy data loaders
    train_loader, val_loader = create_dummy_data_loaders()
    
    # Initialize controller
    vocab_size = 19  # Number of architecture actions
    controller = NASController(vocab_size=vocab_size)
    
    # Initialize training loop
    nas_trainer = NASTrainingLoop(
        controller=controller,
        train_loader=train_loader,
        val_loader=val_loader,
        num_episodes=20,  # Small number for demo
        child_epochs=5,   # Small number for demo
        max_layers=8
    )
    
    # Run training
    results = nas_trainer.run_training()
    
    # Plot results
    nas_trainer.plot_training_progress(results)
    
    # Save results
    nas_trainer.save_results(results)
