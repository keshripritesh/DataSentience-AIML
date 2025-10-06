"""
Monte Carlo Tree Search (MCTS) implementation for NeuralDicePredictor.

This module provides a sophisticated MCTS algorithm that combines
tree search with neural network evaluation for optimal decision making.
"""

import math
import random
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from copy import deepcopy

from ..core.game_state import GameState, GamePhase, DiceAction
from ..core.game_engine import GameEngine


@dataclass
class MCTSConfig:
    """Configuration for MCTS algorithm."""
    simulation_count: int = 1000
    exploration_constant: float = 1.414  # UCB1 constant
    temperature: float = 1.0
    max_depth: int = 100
    use_neural_network: bool = True
    neural_network_confidence: float = 0.8


class MCTSNode:
    """Node in the Monte Carlo Tree Search tree."""
    
    def __init__(self, game_state: GameState, parent: Optional['MCTSNode'] = None, 
                 action: Optional[Tuple] = None):
        self.game_state = game_state
        self.parent = parent
        self.action = action  # (action_type, action_data)
        
        # MCTS statistics
        self.visit_count = 0
        self.total_value = 0.0
        self.children: List['MCTSNode'] = []
        self.untried_actions: List[Tuple] = []
        
        # Neural network evaluation (if available)
        self.neural_value = None
        self.neural_policy = None
        
        # Initialize untried actions
        self._initialize_untried_actions()
    
    def _initialize_untried_actions(self):
        """Initialize list of untried actions from this node."""
        game_engine = GameEngine()
        valid_actions = game_engine.get_valid_actions(self.game_state)
        
        self.untried_actions = []
        for action in valid_actions:
            if action == DiceAction.SCORE:
                self.untried_actions.append((action, None))
            elif action == DiceAction.REROLL:
                self.untried_actions.append((action, {}))
            elif action == DiceAction.KEEP:
                # Generate different keep combinations
                current_dice = self.game_state.current_player_state.dice_state
                available_indices = [i for i in range(len(current_dice.values)) 
                                  if i not in current_dice.kept]
                
                # Add some keep combinations (limit to avoid explosion)
                if len(available_indices) <= 3:
                    for i in range(1, len(available_indices) + 1):
                        for indices in self._get_combinations(available_indices, i):
                            self.untried_actions.append((action, {'dice_indices': indices}))
                else:
                    # Just add a few representative combinations
                    for i in range(1, min(4, len(available_indices))):
                        sample_indices = random.sample(available_indices, i)
                        self.untried_actions.append((action, {'dice_indices': sample_indices}))
    
    def _get_combinations(self, items: List, r: int) -> List[List]:
        """Get all combinations of items of size r."""
        if r == 0:
            return [[]]
        if r > len(items):
            return []
        
        result = []
        for i in range(len(items)):
            for combo in self._get_combinations(items[i+1:], r-1):
                result.append([items[i]] + combo)
        return result
    
    def is_terminal(self) -> bool:
        """Check if this node represents a terminal game state."""
        return self.game_state.is_game_over
    
    def is_fully_expanded(self) -> bool:
        """Check if all actions have been tried from this node."""
        return len(self.untried_actions) == 0
    
    def get_ucb_value(self, exploration_constant: float) -> float:
        """Calculate UCB1 value for this node."""
        if self.visit_count == 0:
            return float('inf')
        
        exploitation = self.total_value / self.visit_count
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visit_count) / self.visit_count
        )
        
        return exploitation + exploration
    
    def select_child(self, exploration_constant: float) -> 'MCTSNode':
        """Select child node using UCB1."""
        if not self.children:
            raise ValueError("No children to select from")
        
        best_child = max(self.children, 
                        key=lambda child: child.get_ucb_value(exploration_constant))
        return best_child
    
    def expand(self) -> 'MCTSNode':
        """Expand this node by trying an untried action."""
        if not self.untried_actions:
            raise ValueError("No untried actions to expand")
        
        # Select random untried action
        action, action_data = random.choice(self.untried_actions)
        self.untried_actions.remove((action, action_data))
        
        # Execute action to get new game state
        game_engine = GameEngine()
        try:
            new_game_state = game_engine.execute_action(self.game_state, action, action_data)
        except Exception:
            # If action fails, create a copy of current state
            new_game_state = self.game_state.copy()
        
        # Create child node
        child = MCTSNode(new_game_state, parent=self, action=(action, action_data))
        self.children.append(child)
        
        return child
    
    def simulate(self, max_depth: int = 100) -> float:
        """Simulate a random game from this node."""
        current_state = self.game_state.copy()
        game_engine = GameEngine()
        depth = 0
        
        while not current_state.is_game_over and depth < max_depth:
            valid_actions = game_engine.get_valid_actions(current_state)
            if not valid_actions:
                break
            
            # Choose random action
            action = random.choice(valid_actions)
            
            try:
                if action == DiceAction.SCORE:
                    current_state = game_engine.execute_action(current_state, action)
                elif action == DiceAction.REROLL:
                    current_state = game_engine.execute_action(current_state, action, {})
                elif action == DiceAction.KEEP:
                    # Randomly choose dice to keep
                    current_dice = current_state.current_player_state.dice_state
                    available_indices = [i for i in range(len(current_dice.values)) 
                                      if i not in current_dice.kept]
                    if available_indices:
                        keep_count = random.randint(1, len(available_indices))
                        keep_indices = random.sample(available_indices, keep_count)
                        current_state = game_engine.execute_action(current_state, action, 
                                                               {'dice_indices': keep_indices})
                    else:
                        break
            except Exception:
                break
            
            depth += 1
        
        # Calculate final score (normalized)
        if current_state.is_game_over:
            winner = current_state.winner
            if winner is not None:
                # Return 1.0 for win, 0.0 for loss
                return 1.0 if winner == 0 else 0.0
            else:
                # Tie game
                return 0.5
        
        # Game didn't finish, return score-based evaluation
        current_player_score = current_state.current_player_state.score
        max_score = max(p.score for p in current_state.players)
        return current_player_score / max(max_score, 1)
    
    def backpropagate(self, value: float):
        """Backpropagate simulation result up the tree."""
        node = self
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            node = node.parent
    
    def get_best_action(self, temperature: float = 1.0) -> Tuple:
        """Get the best action based on visit counts."""
        if not self.children:
            raise ValueError("No children to select best action from")
        
        if temperature == 0:
            # Greedy selection
            best_child = max(self.children, key=lambda child: child.visit_count)
            return best_child.action
        
        # Temperature-scaled selection
        visit_counts = np.array([child.visit_count for child in self.children])
        if temperature > 0:
            # Apply temperature scaling
            scaled_counts = visit_counts ** (1.0 / temperature)
            probabilities = scaled_counts / scaled_counts.sum()
        else:
            # Uniform distribution
            probabilities = np.ones(len(visit_counts)) / len(visit_counts)
        
        # Sample action based on probabilities
        chosen_index = np.random.choice(len(self.children), p=probabilities)
        return self.children[chosen_index].action


class AdvancedMCTS:
    """Advanced Monte Carlo Tree Search with neural network integration."""
    
    def __init__(self, config: MCTSConfig, neural_agent=None):
        self.config = config
        self.neural_agent = neural_agent
        self.game_engine = GameEngine()
    
    def search(self, game_state: GameState) -> Tuple:
        """
        Perform MCTS search to find best action.
        
        Args:
            game_state: Current game state
            
        Returns:
            Best action tuple
        """
        root = MCTSNode(game_state)
        
        # If neural network is available, use it for initial evaluation
        if self.config.use_neural_network and self.neural_agent is not None:
            self._evaluate_node_with_neural_network(root)
        
        # Perform simulations
        for _ in range(self.config.simulation_count):
            node = root
            
            # Selection
            while node.is_fully_expanded() and not node.is_terminal():
                node = node.select_child(self.config.exploration_constant)
            
            # Expansion
            if not node.is_terminal():
                node = node.expand()
                
                # Evaluate new node with neural network if available
                if self.config.use_neural_network and self.neural_agent is not None:
                    self._evaluate_node_with_neural_network(node)
            
            # Simulation
            if node.is_terminal():
                value = self._evaluate_terminal_state(node.game_state)
            else:
                value = node.simulate(self.config.max_depth)
            
            # Backpropagation
            node.backpropagate(value)
        
        # Return best action
        return root.get_best_action(self.config.temperature)
    
    def _evaluate_node_with_neural_network(self, node: MCTSNode):
        """Evaluate node using neural network if available."""
        if self.neural_agent is None:
            return
        
        try:
            # Convert game state to tensor
            state_tensor = node.game_state.to_tensor()
            
            # Get neural network evaluation
            policy_probs = self.neural_agent.get_action_probabilities(state_tensor)
            value = self.neural_agent.network.get_value(
                torch.FloatTensor(state_tensor).unsqueeze(0)
            ).item()
            
            # Store neural network evaluation
            node.neural_policy = policy_probs
            node.neural_value = value
            
        except Exception as e:
            # If neural network evaluation fails, continue without it
            print(f"Neural network evaluation failed: {e}")
    
    def _evaluate_terminal_state(self, game_state: GameState) -> float:
        """Evaluate terminal game state."""
        if not game_state.is_game_over:
            return 0.5
        
        winner = game_state.winner
        if winner is not None:
            return 1.0 if winner == 0 else 0.0
        else:
            # Tie game
            return 0.5
    
    def get_search_statistics(self, root_node: MCTSNode) -> Dict[str, Any]:
        """Get statistics about the MCTS search."""
        if not root_node.children:
            return {}
        
        stats = {
            'total_simulations': root_node.visit_count,
            'children_count': len(root_node.children),
            'action_distribution': {},
            'neural_network_used': self.config.use_neural_network
        }
        
        # Action distribution
        for child in root_node.children:
            action_str = str(child.action)
            stats['action_distribution'][action_str] = {
                'visit_count': child.visit_count,
                'average_value': child.total_value / max(child.visit_count, 1),
                'ucb_value': child.get_ucb_value(self.config.exploration_constant)
            }
        
        return stats


class MCTSPlayer:
    """Player that uses MCTS for decision making."""
    
    def __init__(self, player_id: int, mcts: AdvancedMCTS):
        self.player_id = player_id
        self.mcts = mcts
        self.name = f"MCTS_Player_{player_id}"
    
    def select_action(self, game_state: GameState) -> Tuple:
        """Select action using MCTS."""
        return self.mcts.search(game_state)
    
    def get_action_with_confidence(self, game_state: GameState) -> Tuple[Tuple, float]:
        """Select action and return confidence level."""
        action = self.mcts.search(game_state)
        
        # Calculate confidence based on visit count distribution
        root = MCTSNode(game_state)
        # Note: This is a simplified confidence calculation
        # In practice, you'd want to run a full search and analyze the results
        
        return action, 0.8  # Placeholder confidence value
