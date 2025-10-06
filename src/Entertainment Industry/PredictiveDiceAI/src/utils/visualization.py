"""
Visualization utilities for NeuralDicePredictor.

This module provides comprehensive visualization tools for
analyzing training performance, game statistics, and AI behavior.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class PerformanceAnalyzer:
    """Analyzer for training performance and game statistics."""
    
    def __init__(self, style: str = "seaborn-v0_8"):
        """Initialize the performance analyzer."""
        self.style = style
        plt.style.use(style)
        sns.set_palette("husl")
        
        # Default figure size
        self.figsize = (12, 8)
    
    def plot_training_curves(self, training_stats: Dict[str, List], 
                            save_path: Optional[str] = None):
        """
        Plot training curves including loss and win rate.
        
        Args:
            training_stats: Dictionary containing training statistics
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('NeuralDicePredictor Training Progress', fontsize=16, fontweight='bold')
        
        # Loss curves
        if 'losses' in training_stats and training_stats['losses']:
            losses = training_stats['losses']
            episodes = range(len(losses))
            
            # Total loss
            total_losses = [loss.get('total_loss', 0) for loss in losses]
            axes[0, 0].plot(episodes, total_losses, 'b-', linewidth=2, label='Total Loss')
            axes[0, 0].set_title('Total Loss Over Time')
            axes[0, 0].set_xlabel('Training Step')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Policy and value losses
            policy_losses = [loss.get('policy_loss', 0) for loss in losses]
            value_losses = [loss.get('value_loss', 0) for loss in losses]
            
            axes[0, 1].plot(episodes, policy_losses, 'g-', linewidth=2, label='Policy Loss')
            axes[0, 1].plot(episodes, value_losses, 'r-', linewidth=2, label='Value Loss')
            axes[0, 1].set_title('Policy vs Value Loss')
            axes[0, 1].set_xlabel('Training Step')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Win rate
        if 'win_rates' in training_stats and training_stats['win_rates']:
            win_rates = training_stats['win_rates']
            eval_episodes = range(0, len(win_rates) * 100, 100)  # Assuming evaluation every 100 episodes
            
            axes[1, 0].plot(eval_episodes, win_rates, 'purple', linewidth=2, marker='o')
            axes[1, 0].set_title('Win Rate Over Time')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Win Rate')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_ylim(0, 1)
        
        # Curriculum difficulty
        if 'curriculum_difficulty' in training_stats and training_stats['curriculum_difficulty']:
            difficulties = training_stats['curriculum_difficulty']
            eval_episodes = range(0, len(difficulties) * 100, 100)
            
            axes[1, 1].plot(eval_episodes, difficulties, 'orange', linewidth=2, marker='s')
            axes[1, 1].set_title('Curriculum Difficulty Progression')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Difficulty Level')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves plot saved to {save_path}")
        
        plt.show()
    
    def plot_game_statistics(self, game_stats: List[Dict], save_path: Optional[str] = None):
        """
        Plot game statistics and analysis.
        
        Args:
            game_stats: List of game statistics dictionaries
            save_path: Optional path to save the plot
        """
        if not game_stats:
            print("No game statistics to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Game Statistics Analysis', fontsize=16, fontweight='bold')
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(game_stats)
        
        # Score distribution
        if 'final_scores' in df.columns:
            all_scores = []
            for scores in df['final_scores']:
                if scores:
                    all_scores.extend(scores)
            
            if all_scores:
                axes[0, 0].hist(all_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                axes[0, 0].set_title('Final Score Distribution')
                axes[0, 0].set_xlabel('Score')
                axes[0, 0].set_ylabel('Frequency')
                axes[0, 0].grid(True, alpha=0.3)
        
        # Game length distribution
        if 'turn_number' in df.columns:
            axes[0, 1].hist(df['turn_number'], bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[0, 1].set_title('Game Length Distribution')
            axes[0, 1].set_xlabel('Number of Turns')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Win rate by player
        if 'winner' in df.columns:
            winner_counts = df['winner'].value_counts()
            if not winner_counts.empty:
                axes[1, 0].pie(winner_counts.values, labels=[f'Player {i+1}' for i in winner_counts.index], 
                              autopct='%1.1f%%', startangle=90)
                axes[1, 0].set_title('Win Distribution by Player')
        
        # Score progression over turns
        if 'game_history' in df.columns:
            # Analyze score progression (simplified)
            turn_scores = []
            for game in game_stats:
                if 'game_history' in game and game['game_history']:
                    for event in game['game_history']:
                        if event.get('type') == 'score':
                            turn_scores.append(event.get('points', 0))
            
            if turn_scores:
                axes[1, 1].hist(turn_scores, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
                axes[1, 1].set_title('Score per Turn Distribution')
                axes[1, 1].set_xlabel('Points Scored')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Game statistics plot saved to {save_path}")
        
        plt.show()
    
    def plot_mcts_analysis(self, mcts_stats: Dict[str, Any], save_path: Optional[str] = None):
        """
        Plot MCTS search analysis.
        
        Args:
            mcts_stats: MCTS search statistics
            save_path: Optional path to save the plot
        """
        if not mcts_stats:
            print("No MCTS statistics to plot")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('MCTS Search Analysis', fontsize=16, fontweight='bold')
        
        # Action distribution
        if 'action_distribution' in mcts_stats:
            actions = list(mcts_stats['action_distribution'].keys())
            visit_counts = [mcts_stats['action_distribution'][action]['visit_count'] 
                          for action in actions]
            
            # Truncate long action names for display
            display_actions = [action[:20] + '...' if len(action) > 20 else action 
                             for action in actions]
            
            bars = axes[0].bar(range(len(actions)), visit_counts, color='lightblue', alpha=0.7)
            axes[0].set_title('MCTS Action Visit Distribution')
            axes[0].set_xlabel('Actions')
            axes[0].set_ylabel('Visit Count')
            axes[0].set_xticks(range(len(actions)))
            axes[0].set_xticklabels(display_actions, rotation=45, ha='right')
            axes[0].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, count in zip(bars, visit_counts):
                height = bar.get_height()
                axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{count}', ha='center', va='bottom')
        
        # UCB values
        if 'action_distribution' in mcts_stats:
            ucb_values = [mcts_stats['action_distribution'][action]['ucb_value'] 
                         for action in actions]
            
            axes[1].bar(range(len(actions)), ucb_values, color='lightgreen', alpha=0.7)
            axes[1].set_title('MCTS UCB Values')
            axes[1].set_xlabel('Actions')
            axes[1].set_ylabel('UCB Value')
            axes[1].set_xticks(range(len(actions)))
            axes[1].set_xticklabels(display_actions, rotation=45, ha='right')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"MCTS analysis plot saved to {save_path}")
        
        plt.show()
    
    def create_interactive_dashboard(self, training_stats: Dict[str, List], 
                                   game_stats: List[Dict]):
        """
        Create an interactive Plotly dashboard.
        
        Args:
            training_stats: Training statistics
            game_stats: Game statistics
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Training Loss', 'Win Rate', 'Score Distribution', 
                          'Game Length', 'Curriculum Difficulty', 'MCTS Analysis'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Training loss
        if 'losses' in training_stats and training_stats['losses']:
            losses = training_stats['losses']
            episodes = list(range(len(losses)))
            total_losses = [loss.get('total_loss', 0) for loss in losses]
            
            fig.add_trace(
                go.Scatter(x=episodes, y=total_losses, mode='lines', name='Total Loss',
                          line=dict(color='blue', width=2)),
                row=1, col=1
            )
        
        # Win rate
        if 'win_rates' in training_stats and training_stats['win_rates']:
            win_rates = training_stats['win_rates']
            eval_episodes = list(range(0, len(win_rates) * 100, 100))
            
            fig.add_trace(
                go.Scatter(x=eval_episodes, y=win_rates, mode='lines+markers', 
                          name='Win Rate', line=dict(color='purple', width=2)),
                row=1, col=2
            )
        
        # Score distribution
        if game_stats:
            df = pd.DataFrame(game_stats)
            if 'final_scores' in df.columns:
                all_scores = []
                for scores in df['final_scores']:
                    if scores:
                        all_scores.extend(scores)
                
                if all_scores:
                    fig.add_trace(
                        go.Histogram(x=all_scores, name='Score Distribution',
                                   marker_color='skyblue', opacity=0.7),
                        row=2, col=1
                    )
        
        # Game length
        if game_stats:
            df = pd.DataFrame(game_stats)
            if 'turn_number' in df.columns:
                fig.add_trace(
                    go.Histogram(x=df['turn_number'], name='Game Length',
                               marker_color='lightgreen', opacity=0.7),
                    row=2, col=2
                )
        
        # Curriculum difficulty
        if 'curriculum_difficulty' in training_stats and training_stats['curriculum_difficulty']:
            difficulties = training_stats['curriculum_difficulty']
            eval_episodes = list(range(0, len(difficulties) * 100, 100))
            
            fig.add_trace(
                go.Scatter(x=eval_episodes, y=difficulties, mode='lines+markers',
                          name='Difficulty', line=dict(color='orange', width=2)),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            title_text="NeuralDicePredictor Interactive Dashboard",
            showlegend=True,
            height=900
        )
        
        # Show the dashboard
        fig.show()
    
    def plot_action_heatmap(self, action_history: List[Dict], save_path: Optional[str] = None):
        """
        Plot action selection heatmap.
        
        Args:
            action_history: List of action history dictionaries
            save_path: Optional path to save the plot
        """
        if not action_history:
            print("No action history to plot")
            return
        
        # Create action matrix
        actions = ['Score', 'Reroll', 'Keep']
        game_states = []
        action_counts = []
        
        for game in action_history:
            if 'actions' in game:
                game_actions = game['actions']
                state_actions = [0, 0, 0]  # [Score, Reroll, Keep]
                
                for action in game_actions:
                    if 'score' in action.lower():
                        state_actions[0] += 1
                    elif 'reroll' in action.lower():
                        state_actions[1] += 1
                    elif 'keep' in action.lower():
                        state_actions[2] += 1
                
                action_counts.append(state_actions)
                game_states.append(f"Game {len(game_states) + 1}")
        
        if not action_counts:
            print("No valid action data found")
            return
        
        # Create heatmap
        action_matrix = np.array(action_counts).T
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(action_matrix, 
                   xticklabels=game_states, 
                   yticklabels=actions,
                   annot=True, 
                   fmt='d', 
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Action Count'})
        
        plt.title('Action Selection Heatmap', fontsize=16, fontweight='bold')
        plt.xlabel('Games', fontsize=12)
        plt.ylabel('Actions', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Action heatmap saved to {save_path}")
        
        plt.show()
    
    def save_training_report(self, training_stats: Dict[str, List], 
                           game_stats: List[Dict], filepath: str):
        """
        Save a comprehensive training report.
        
        Args:
            training_stats: Training statistics
            game_stats: Game statistics
            filepath: Path to save the report
        """
        with open(filepath, 'w') as f:
            f.write("NeuralDicePredictor Training Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Training summary
            f.write("Training Summary:\n")
            f.write("-" * 20 + "\n")
            
            if 'losses' in training_stats and training_stats['losses']:
                final_loss = training_stats['losses'][-1]
                f.write(f"Final Total Loss: {final_loss.get('total_loss', 'N/A'):.6f}\n")
                f.write(f"Final Policy Loss: {final_loss.get('policy_loss', 'N/A'):.6f}\n")
                f.write(f"Final Value Loss: {final_loss.get('value_loss', 'N/A'):.6f}\n")
            
            if 'win_rates' in training_stats and training_stats['win_rates']:
                win_rates = training_stats['win_rates']
                f.write(f"Final Win Rate: {win_rates[-1]:.3f}\n")
                f.write(f"Best Win Rate: {max(win_rates):.3f}\n")
                f.write(f"Average Win Rate: {np.mean(win_rates):.3f}\n")
            
            # Game statistics
            f.write("\nGame Statistics:\n")
            f.write("-" * 20 + "\n")
            
            if game_stats:
                df = pd.DataFrame(game_stats)
                f.write(f"Total Games: {len(game_stats)}\n")
                
                if 'winner' in df.columns:
                    winner_counts = df['winner'].value_counts()
                    f.write("Win Distribution:\n")
                    for winner, count in winner_counts.items():
                        f.write(f"  Player {winner + 1}: {count} wins\n")
                
                if 'turn_number' in df.columns:
                    f.write(f"Average Game Length: {df['turn_number'].mean():.1f} turns\n")
                    f.write(f"Min Game Length: {df['turn_number'].min()} turns\n")
                    f.write(f"Max Game Length: {df['turn_number'].max()} turns\n")
        
        print(f"Training report saved to {filepath}")


def create_sample_visualizations():
    """Create sample visualizations for demonstration."""
    analyzer = PerformanceAnalyzer()
    
    # Sample training statistics
    sample_training_stats = {
        'losses': [
            {'total_loss': 2.5, 'policy_loss': 1.2, 'value_loss': 1.3},
            {'total_loss': 2.1, 'policy_loss': 1.0, 'value_loss': 1.1},
            {'total_loss': 1.8, 'policy_loss': 0.8, 'value_loss': 1.0},
            {'total_loss': 1.5, 'policy_loss': 0.7, 'value_loss': 0.8},
            {'total_loss': 1.2, 'policy_loss': 0.6, 'value_loss': 0.6}
        ],
        'win_rates': [0.3, 0.4, 0.5, 0.6, 0.7],
        'curriculum_difficulty': [0.1, 0.2, 0.3, 0.4, 0.5]
    }
    
    # Sample game statistics
    sample_game_stats = [
        {'turn_number': 8, 'winner': 0, 'final_scores': [850, 720]},
        {'turn_number': 12, 'winner': 1, 'final_scores': [650, 920]},
        {'turn_number': 10, 'winner': 0, 'final_scores': [780, 690]},
        {'turn_number': 15, 'winner': 1, 'final_scores': [580, 1100]},
        {'turn_number': 9, 'winner': 0, 'final_scores': [890, 650]}
    ]
    
    # Create visualizations
    analyzer.plot_training_curves(sample_training_stats)
    analyzer.plot_game_statistics(sample_game_stats)
    
    print("Sample visualizations created successfully!")


if __name__ == "__main__":
    create_sample_visualizations()
