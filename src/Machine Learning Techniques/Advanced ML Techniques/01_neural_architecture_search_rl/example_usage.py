"""
Complete Working Example: Neural Architecture Search with Reinforcement Learning
This script demonstrates the full NAS pipeline from start to finish.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import time
import json

# Import our NAS components
from nas_controller import NASController, PolicyGradientTrainer, decode_architecture, ARCHITECTURE_VOCAB
from child_network import ChildNetwork, ChildNetworkTrainer, build_child_network_from_actions
from training_loop import NASTrainingLoop, create_dummy_data_loaders


def demonstrate_controller():
    """Demonstrate the NAS controller functionality."""
    print("🔧 Demonstrating NAS Controller")
    print("=" * 50)
    
    # Initialize controller
    vocab_size = len(ARCHITECTURE_VOCAB)
    controller = NASController(vocab_size=vocab_size, hidden_size=64)
    
    print(f"Controller initialized with vocabulary size: {vocab_size}")
    print(f"Available actions: {list(ARCHITECTURE_VOCAB.values())}")
    
    # Sample architectures
    print("\n📊 Sampling architectures from controller:")
    for i in range(3):
        architecture = controller.sample_architecture(max_layers=6, temperature=1.0)
        decoded = decode_architecture(architecture)
        print(f"Architecture {i+1}: {decoded}")
    
    # Test policy gradient update
    print("\n🔄 Testing policy gradient update:")
    trainer = PolicyGradientTrainer(controller, lr=0.001)
    
    # Simulate some experiences
    actions_list = [
        [2, 14, 12, 5, 14],  # CONV_3x3, RELU, BATCH_NORM, MAX_POOL, RELU
        [3, 14, 12, 6, 14],  # CONV_5x5, RELU, BATCH_NORM, AVG_POOL, RELU
        [7, 14, 12, 5, 14],  # SEPARABLE_CONV_3x3, RELU, BATCH_NORM, MAX_POOL, RELU
    ]
    rewards = [0.85, 0.78, 0.92]  # Simulated rewards
    
    loss = trainer.update_policy(actions_list, rewards)
    print(f"Policy gradient loss: {loss:.4f}")
    
    return controller


def demonstrate_child_network():
    """Demonstrate the child network functionality."""
    print("\n🏗️ Demonstrating Child Network Builder")
    print("=" * 50)
    
    # Define a sample architecture
    actions = [2, 14, 12, 5, 14, 3, 14, 12]  # CONV_3x3, RELU, BATCH_NORM, MAX_POOL, RELU, CONV_5x5, RELU, BATCH_NORM
    
    # Build child network
    input_shape = (3, 32, 32)
    num_classes = 10
    child_net = build_child_network_from_actions(actions, input_shape, num_classes)
    
    print(f"Child network built successfully!")
    print(f"Input shape: {input_shape}")
    print(f"Number of classes: {num_classes}")
    print(f"Number of parameters: {child_net.get_num_parameters():,}")
    print(f"Model size: {child_net.get_model_size_mb():.2f} MB")
    
    # Test forward pass
    x = torch.randn(2, *input_shape)  # Batch of 2 samples
    with torch.no_grad():
        output = child_net(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output logits: {output[0][:5]}...")  # First 5 logits of first sample
    
    return child_net


def demonstrate_training_loop():
    """Demonstrate the complete training loop."""
    print("\n🚀 Demonstrating Complete Training Loop")
    print("=" * 50)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_dummy_data_loaders()
    
    # Initialize controller
    vocab_size = len(ARCHITECTURE_VOCAB)
    controller = NASController(vocab_size=vocab_size, hidden_size=64)
    
    # Initialize training loop
    nas_trainer = NASTrainingLoop(
        controller=controller,
        train_loader=train_loader,
        val_loader=val_loader,
        num_episodes=5,    # Small number for demo
        child_epochs=3,    # Small number for demo
        max_layers=6
    )
    
    print("Starting NAS training...")
    print(f"Episodes: {nas_trainer.num_episodes}")
    print(f"Child epochs per episode: {nas_trainer.child_epochs}")
    print(f"Max layers per architecture: {nas_trainer.max_layers}")
    
    # Run training
    results = nas_trainer.run_training()
    
    print(f"\n🎯 Training completed!")
    print(f"Best reward: {results['best_reward']:.4f}")
    print(f"Best architecture: {decode_architecture(results['best_architecture'])}")
    
    return results


def analyze_results(results: Dict):
    """Analyze and visualize the training results."""
    print("\n📈 Analyzing Training Results")
    print("=" * 50)
    
    # Basic statistics
    rewards = results['episode_rewards']
    accuracies = results['episode_accuracies']
    
    print(f"Number of episodes: {len(rewards)}")
    print(f"Average reward: {np.mean(rewards):.4f}")
    print(f"Best reward: {np.max(rewards):.4f}")
    print(f"Average accuracy: {np.mean(accuracies):.4f}")
    print(f"Best accuracy: {np.max(accuracies):.4f}")
    
    # Plot results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    ax1.plot(rewards, 'b-', alpha=0.7)
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True, alpha=0.3)
    
    # Episode accuracies
    ax2.plot(accuracies, 'g-', alpha=0.7)
    ax2.set_title('Episode Accuracies')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True, alpha=0.3)
    
    # Moving average rewards
    window = min(5, len(rewards))
    if window > 1:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax3.plot(range(window-1, len(rewards)), moving_avg, 'r-', linewidth=2)
        ax3.set_title(f'Moving Average Reward (window={window})')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Average Reward')
        ax3.grid(True, alpha=0.3)
    
    # Reward distribution
    ax4.hist(rewards, bins=min(10, len(rewards)), alpha=0.7, color='skyblue', edgecolor='black')
    ax4.set_title('Reward Distribution')
    ax4.set_xlabel('Reward')
    ax4.set_ylabel('Frequency')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('nas_demo_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save results
    with open('nas_demo_results.json', 'w') as f:
        json.dump({
            'episode_rewards': [float(r) for r in rewards],
            'episode_accuracies': [float(a) for a in accuracies],
            'best_reward': float(results['best_reward']),
            'best_architecture': results['best_architecture'],
            'total_time': float(results['total_time'])
        }, f, indent=2)
    
    print("Results saved to 'nas_demo_results.json'")
    print("Plots saved to 'nas_demo_results.png'")


def demonstrate_advanced_features():
    """Demonstrate advanced NAS features."""
    print("\n🔬 Demonstrating Advanced Features")
    print("=" * 50)
    
    # Temperature scaling for exploration vs exploitation
    controller = NASController(vocab_size=len(ARCHITECTURE_VOCAB))
    
    print("🌡️ Temperature scaling demonstration:")
    temperatures = [0.5, 1.0, 2.0]
    
    for temp in temperatures:
        print(f"\nTemperature: {temp}")
        architecture = controller.sample_architecture(max_layers=5, temperature=temp)
        decoded = decode_architecture(architecture)
        print(f"Architecture: {decoded}")
    
    # Architecture complexity analysis
    print("\n📊 Architecture complexity analysis:")
    architectures = [
        [2, 14, 12],           # Simple: CONV_3x3, RELU, BATCH_NORM
        [2, 14, 12, 5, 14, 3], # Medium: CONV_3x3, RELU, BATCH_NORM, MAX_POOL, RELU, CONV_5x5
        [2, 14, 12, 5, 14, 3, 14, 12, 7, 14, 12, 6]  # Complex: Multiple layers
    ]
    
    for i, actions in enumerate(architectures):
        child_net = build_child_network_from_actions(actions, (3, 32, 32), 10)
        print(f"Architecture {i+1}: {len(actions)} layers, {child_net.get_num_parameters():,} parameters")


def main():
    """Main demonstration function."""
    print("🧠 Neural Architecture Search with Reinforcement Learning - Complete Demo")
    print("=" * 80)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # Demonstrate each component
        controller = demonstrate_controller()
        child_net = demonstrate_child_network()
        results = demonstrate_training_loop()
        
        # Analyze results
        analyze_results(results)
        
        # Demonstrate advanced features
        demonstrate_advanced_features()
        
        print("\n✅ Demo completed successfully!")
        print("\n📁 Generated files:")
        print("- nas_demo_results.json: Training results")
        print("- nas_demo_results.png: Training plots")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
