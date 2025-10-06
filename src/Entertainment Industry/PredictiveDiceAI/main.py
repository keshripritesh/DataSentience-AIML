#!/usr/bin/env python3
"""
Main execution script for NeuralDicePredictor.

This script provides the main entry point for training, evaluation,
and interactive gameplay with the AI agent.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.game_engine import GameEngine
from core.game_state import GameState, GamePhase
from ai.neural_net import NeuralAgent, NetworkConfig
from ai.mcts import AdvancedMCTS, MCTSConfig, MCTSPlayer
from ai.training import TrainingPipeline, TrainingConfig
from utils.visualization import PerformanceAnalyzer


def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = ["models", "logs", "config"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)


def train_agent(args):
    """Train the neural network agent."""
    print("🚀 Starting NeuralDicePredictor training...")
    
    # Setup directories
    setup_directories()
    
    # Training configuration
    config = TrainingConfig(
        num_episodes=args.episodes,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        experience_buffer_size=args.buffer_size,
        self_play_games_per_update=args.games_per_update,
        evaluation_interval=args.eval_interval,
        save_interval=args.save_interval,
        curriculum_learning=args.curriculum,
        temperature_decay=args.temp_decay,
        min_temperature=args.min_temp
    )
    
    # Initialize training pipeline
    pipeline = TrainingPipeline(config)
    
    # Start training
    try:
        pipeline.train()
        
        # Get training summary
        summary = pipeline.get_training_summary()
        print("\n🎯 Training Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # Save final model
        pipeline._save_final_model()
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        pipeline._save_final_model()
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise


def evaluate_agent(args):
    """Evaluate a trained agent."""
    print("🔍 Evaluating trained agent...")
    
    if not args.model_path:
        print("❌ Model path is required for evaluation")
        return
    
    # Load trained agent
    try:
        network_config = NetworkConfig(
            input_size=50,
            hidden_sizes=(256, 256, 128),
            output_size=3,
            learning_rate=0.001
        )
        
        agent = NeuralAgent(network_config)
        agent.load_model(args.model_path)
        print(f"✅ Model loaded from {args.model_path}")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Initialize MCTS with neural agent
    mcts_config = MCTSConfig(
        simulation_count=args.simulations,
        exploration_constant=1.414,
        temperature=args.temperature,
        use_neural_network=True
    )
    
    mcts = AdvancedMCTS(mcts_config, agent)
    
    # Evaluate against random opponent
    game_engine = GameEngine()
    wins = 0
    total_games = args.eval_games
    
    print(f"🎲 Playing {total_games} evaluation games...")
    
    for game_num in range(total_games):
        # Create game
        game_state = game_engine.create_initial_state(
            num_players=2,
            max_turns=args.max_turns
        )
        
        # Play game
        while not game_state.is_game_over:
            if game_state.current_player == 0:
                # AI agent's turn
                action, action_data = mcts.search(game_state)
            else:
                # Random opponent's turn
                valid_actions = game_engine.get_valid_actions(game_state)
                if not valid_actions:
                    break
                
                import random
                action = random.choice(valid_actions)
                action_data = {} if action.value != "keep" else {'dice_indices': [0]}
            
            try:
                game_state = game_engine.execute_action(game_state, action, action_data)
            except Exception as e:
                print(f"Game {game_num + 1} failed: {e}")
                break
        
        # Check winner
        if game_state.is_game_over and game_state.winner == 0:
            wins += 1
        
        if (game_num + 1) % 10 == 0:
            print(f"  Games played: {game_num + 1}, Wins: {wins}")
    
    win_rate = wins / total_games
    print(f"\n🎯 Evaluation Results:")
    print(f"  Total Games: {total_games}")
    print(f"  Wins: {wins}")
    print(f"  Win Rate: {win_rate:.3f} ({win_rate*100:.1f}%)")


def play_interactive(args):
    """Play interactive game against AI."""
    print("🎮 Starting interactive game...")
    
    if not args.model_path:
        print("❌ Model path is required for interactive play")
        return
    
    # Load trained agent
    try:
        network_config = NetworkConfig(
            input_size=50,
            hidden_sizes=(256, 256, 128),
            output_size=3,
            learning_rate=0.001
        )
        
        agent = NeuralAgent(network_config)
        agent.load_model(args.model_path)
        print(f"✅ AI agent loaded from {args.model_path}")
        
    except Exception as e:
        print(f"❌ Failed to load AI agent: {e}")
        return
    
    # Initialize MCTS
    mcts_config = MCTSConfig(
        simulation_count=args.simulations,
        exploration_constant=1.414,
        temperature=args.temperature,
        use_neural_network=True
    )
    
    mcts = AdvancedMCTS(mcts_config, agent)
    
    # Initialize game
    game_engine = GameEngine()
    game_state = game_engine.create_initial_state(
        num_players=2,
        max_turns=args.max_turns
    )
    
    print(f"\n🎲 Game started! You are Player 1, AI is Player 2")
    print(f"📊 Target score: {args.target_score}")
    
    while not game_state.is_game_over:
        current_player = game_state.current_player_state
        print(f"\n{'='*50}")
        print(f"🎯 Turn {game_state.turn_number + 1}")
        print(f"👤 Player {game_state.current_player + 1}'s turn")
        print(f"🎲 Dice: {current_player.dice_state.values}")
        print(f"📊 Score: {current_player.score}")
        print(f"🔄 Rerolls used: {current_player.dice_state.reroll_count}")
        
        if game_state.current_player == 0:
            # Human player's turn
            print("\n🤔 Your turn! Choose an action:")
            print("  1. Score dice")
            print("  2. Reroll dice")
            print("  3. Keep dice")
            
            try:
                choice = input("Enter choice (1-3): ").strip()
                
                if choice == "1":
                    # Score dice
                    game_state = game_engine.execute_action(game_state, "score")
                    print("✅ Dice scored!")
                    
                elif choice == "2":
                    # Reroll dice
                    game_state = game_engine.execute_action(game_state, "reroll")
                    print("🔄 Dice rerolled!")
                    
                elif choice == "3":
                    # Keep dice
                    dice_input = input("Enter dice indices to keep (e.g., 0,2,4): ").strip()
                    try:
                        indices = [int(x.strip()) for x in dice_input.split(",")]
                        game_state = game_engine.execute_action(
                            game_state, "keep", {'dice_indices': indices}
                        )
                        print(f"💾 Kept dice at positions: {indices}")
                    except ValueError:
                        print("❌ Invalid dice indices")
                        continue
                else:
                    print("❌ Invalid choice")
                    continue
                    
            except KeyboardInterrupt:
                print("\n👋 Game interrupted")
                return
            except Exception as e:
                print(f"❌ Action failed: {e}")
                continue
        else:
            # AI's turn
            print("\n🤖 AI is thinking...")
            action, action_data = mcts.search(game_state)
            
            try:
                game_state = game_engine.execute_action(game_state, action, action_data)
                print(f"🤖 AI chose: {action}")
            except Exception as e:
                print(f"❌ AI action failed: {e}")
                break
        
        # Check if game is over
        if game_state.is_game_over:
            print(f"\n🏁 Game Over!")
            winner = game_state.winner
            if winner is not None:
                print(f"🎉 Player {winner + 1} wins!")
            else:
                print("🤝 It's a tie!")
            
            print(f"\n📊 Final Scores:")
            for player in game_state.players:
                print(f"  Player {player.player_id + 1}: {player.score}")
            break


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="NeuralDicePredictor - Advanced AI Dice Game",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new agent
  python main.py train --episodes 1000 --batch-size 32
  
  # Evaluate a trained agent
  python main.py evaluate --model-path models/final_model.pt
  
  # Play against AI
  python main.py play --model-path models/final_model.pt
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new AI agent')
    train_parser.add_argument('--episodes', type=int, default=1000,
                            help='Number of training episodes')
    train_parser.add_argument('--batch-size', type=int, default=64,
                            help='Training batch size')
    train_parser.add_argument('--learning-rate', type=float, default=0.001,
                            help='Learning rate')
    train_parser.add_argument('--buffer-size', type=int, default=10000,
                            help='Experience buffer size')
    train_parser.add_argument('--games-per-update', type=int, default=10,
                            help='Self-play games per network update')
    train_parser.add_argument('--eval-interval', type=int, default=100,
                            help='Evaluation interval')
    train_parser.add_argument('--save-interval', type=int, default=500,
                            help='Model save interval')
    train_parser.add_argument('--curriculum', action='store_true',
                            help='Enable curriculum learning')
    train_parser.add_argument('--temp-decay', type=float, default=0.995,
                            help='Temperature decay rate')
    train_parser.add_argument('--min-temp', type=float, default=0.1,
                            help='Minimum temperature')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained agent')
    eval_parser.add_argument('--model-path', type=str, required=True,
                           help='Path to trained model')
    eval_parser.add_argument('--eval-games', type=int, default=100,
                           help='Number of evaluation games')
    eval_parser.add_argument('--simulations', type=int, default=100,
                           help='MCTS simulations per move')
    eval_parser.add_argument('--temperature', type=float, default=1.0,
                           help='MCTS temperature')
    eval_parser.add_argument('--max-turns', type=int, default=10,
                           help='Maximum turns per game')
    
    # Play command
    play_parser = subparsers.add_parser('play', help='Play against AI')
    play_parser.add_argument('--model-path', type=str, required=True,
                           help='Path to trained model')
    play_parser.add_argument('--simulations', type=int, default=100,
                           help='MCTS simulations per move')
    play_parser.add_argument('--temperature', type=float, default=1.0,
                           help='MCTS temperature')
    play_parser.add_argument('--max-turns', type=int, default=10,
                           help='Maximum turns per game')
    play_parser.add_argument('--target-score', type=int, default=1000,
                           help='Target score to win')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    try:
        if args.command == 'train':
            train_agent(args)
        elif args.command == 'evaluate':
            evaluate_agent(args)
        elif args.command == 'play':
            play_interactive(args)
        else:
            print(f"❌ Unknown command: {args.command}")
            
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
