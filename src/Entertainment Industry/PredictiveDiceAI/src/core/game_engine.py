"""
Advanced game engine for NeuralDicePredictor.

This module provides a sophisticated game engine that handles
game logic, scoring rules, turn management, and game progression.
"""

import random
import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass

from .game_state import (
    GameState, PlayerState, DiceState, GamePhase, DiceAction
)


@dataclass
class ScoringRule:
    """Represents a scoring rule for the dice game."""
    name: str
    description: str
    points: int
    condition: Callable[[Tuple[int, ...]], bool]
    bonus_multiplier: float = 1.0
    
    def evaluate(self, dice_values: Tuple[int, ...]) -> int:
        """Evaluate if the rule applies and return points."""
        if self.condition(dice_values):
            return int(self.points * self.bonus_multiplier)
        return 0


class AdvancedScoringEngine:
    """Advanced scoring engine with multiple rule sets."""
    
    def __init__(self):
        self.rules = self._initialize_default_rules()
    
    def _initialize_default_rules(self) -> List[ScoringRule]:
        """Initialize default scoring rules."""
        rules = [
            # Three of a kind
            ScoringRule(
                "Three Ones", "Three 1s = 1000 points", 1000,
                lambda dice: dice.count(1) >= 3
            ),
            ScoringRule(
                "Three Twos", "Three 2s = 200 points", 200,
                lambda dice: dice.count(2) >= 3
            ),
            ScoringRule(
                "Three Threes", "Three 3s = 300 points", 300,
                lambda dice: dice.count(3) >= 3
            ),
            ScoringRule(
                "Three Fours", "Three 4s = 400 points", 400,
                lambda dice: dice.count(4) >= 3
            ),
            ScoringRule(
                "Three Fives", "Three 5s = 500 points", 500,
                lambda dice: dice.count(5) >= 3
            ),
            ScoringRule(
                "Three Sixes", "Three 6s = 600 points", 600,
                lambda dice: dice.count(6) >= 3
            ),
            
            # Special combinations
            ScoringRule(
                "Straight", "1-2-3-4-5-6 = 1500 points", 1500,
                lambda dice: set(dice) == {1, 2, 3, 4, 5, 6}
            ),
            ScoringRule(
                "Three Pairs", "Three pairs = 1500 points", 1500,
                lambda dice: self._has_three_pairs(dice)
            ),
            ScoringRule(
                "Four of a Kind", "Four of any number = 1000 points", 1000,
                lambda dice: any(dice.count(val) >= 4 for val in set(dice))
            ),
            ScoringRule(
                "Five of a Kind", "Five of any number = 2000 points", 2000,
                lambda dice: any(dice.count(val) >= 5 for val in set(dice))
            ),
            ScoringRule(
                "Six of a Kind", "Six of any number = 3000 points", 3000,
                lambda dice: any(dice.count(val) == 6 for val in set(dice))
            ),
        ]
        return rules
    
    def _has_three_pairs(self, dice: Tuple[int, ...]) -> bool:
        """Check if dice contain three pairs."""
        if len(dice) != 6:
            return False
        
        value_counts = {}
        for val in dice:
            value_counts[val] = value_counts.get(val, 0) + 1
        
        # Count pairs (values that appear 2 or more times)
        pair_count = 0
        for count in value_counts.values():
            if count >= 2:
                pair_count += 1
        
        return pair_count >= 3
    
    def calculate_score(self, dice_values: Tuple[int, ...]) -> int:
        """Calculate total score for given dice values."""
        total_score = 0
        
        # Basic scoring (per-die scoring)
        ones_count = dice_values.count(1)
        fives_count = dice_values.count(5)
        total_score += ones_count * 100
        total_score += fives_count * 50
        
        # Apply rule-based scoring
        for rule in self.rules:
            total_score += rule.evaluate(dice_values)
        
        return total_score
    
    def get_available_combinations(self, dice_values: Tuple[int, ...]) -> List[Dict]:
        """Get all available scoring combinations."""
        combinations = []
        for rule in self.rules:
            score = rule.evaluate(dice_values)
            if score > 0:
                combinations.append({
                    'rule': rule.name,
                    'description': rule.description,
                    'points': score,
                    'dice_used': self._get_dice_used_for_rule(rule, dice_values)
                })
        return combinations
    
    def _get_dice_used_for_rule(self, rule: ScoringRule, dice_values: Tuple[int, ...]) -> List[int]:
        """Get indices of dice used for a specific rule."""
        # This is a simplified implementation
        # In practice, you'd want more sophisticated logic
        return list(range(len(dice_values)))


class GameEngine:
    """Advanced game engine for managing dice game progression."""
    
    def __init__(self, num_dice: int = 6, max_rerolls: int = 3):
        self.num_dice = num_dice
        self.max_rerolls = max_rerolls
        self.scoring_engine = AdvancedScoringEngine()
        self.random_seed = None
    
    def set_random_seed(self, seed: int):
        """Set random seed for reproducible games."""
        self.random_seed = seed
        random.seed(seed)
        np.random.seed(seed)
    
    def create_initial_state(self, num_players: int, max_turns: int = 10) -> GameState:
        """Create initial game state."""
        if num_players < 1:
            raise ValueError("Number of players must be at least 1")
        
        # Create initial player states
        players = []
        for i in range(num_players):
            initial_dice = self._roll_dice(self.num_dice)
            dice_state = DiceState(initial_dice)
            player_state = PlayerState(
                player_id=i,
                score=0,
                turn_score=0,
                dice_state=dice_state
            )
            players.append(player_state)
        
        return GameState(
            players=tuple(players),
            current_player=0,
            phase=GamePhase.ROLLING,
            turn_number=0,
            max_turns=max_turns
        )
    
    def _roll_dice(self, count: int) -> Tuple[int, ...]:
        """Roll specified number of dice."""
        return tuple(random.randint(1, 6) for _ in range(count))
    
    def get_valid_actions(self, game_state: GameState) -> List[DiceAction]:
        """Get valid actions for current game state."""
        current_player = game_state.current_player_state
        dice_state = current_player.dice_state
        
        actions = []
        
        # Can always score if there are points
        if self.scoring_engine.calculate_score(dice_state.values) > 0:
            actions.append(DiceAction.SCORE)
        
        # Can reroll if haven't exceeded max rerolls and have dice to reroll
        if (dice_state.reroll_count < self.max_rerolls and 
            len(dice_state.available_dice) > 0):
            actions.append(DiceAction.REROLL)
        
        # Can keep dice if have dice to keep
        if len(dice_state.available_dice) > 0:
            actions.append(DiceAction.KEEP)
        
        return actions
    
    def execute_action(self, game_state: GameState, action: DiceAction, 
                      action_data: Optional[Dict] = None) -> GameState:
        """Execute an action and return new game state."""
        current_player = game_state.current_player_state
        dice_state = current_player.dice_state
        
        if action == DiceAction.SCORE:
            return self._execute_score_action(game_state)
        elif action == DiceAction.REROLL:
            return self._execute_reroll_action(game_state, action_data)
        elif action == DiceAction.KEEP:
            return self._execute_keep_action(game_state, action_data)
        else:
            raise ValueError(f"Unknown action: {action}")
    
    def _execute_score_action(self, game_state: GameState) -> GameState:
        """Execute scoring action."""
        current_player = game_state.current_player_state
        dice_state = current_player.dice_state
        
        # Calculate score
        score = self.scoring_engine.calculate_score(dice_state.values)
        if score == 0:
            raise ValueError("Cannot score dice with no points")
        
        # Update player state
        new_score = current_player.score + score
        new_player_state = current_player.update_score(new_score)
        
        # Add to history
        history_event = {
            'type': 'score',
            'player': current_player.player_id,
            'dice': dice_state.values,
            'points': score,
            'total_score': new_score
        }
        
        # Update the game state with the new player state
        new_game_state = game_state.update_player(
            current_player.player_id, new_player_state
        ).add_to_history(history_event)
        
        # Check if game is over
        if new_game_state.turn_number >= new_game_state.max_turns:
            return new_game_state.change_phase(GamePhase.GAME_OVER)
        
        # Next turn
        return new_game_state.next_turn()
    
    def _execute_reroll_action(self, game_state: GameState, 
                              action_data: Optional[Dict]) -> GameState:
        """Execute reroll action."""
        current_player = game_state.current_player_state
        dice_state = current_player.dice_state
        
        if dice_state.reroll_count >= self.max_rerolls:
            raise ValueError("Maximum rerolls exceeded")
        
        if len(dice_state.available_dice) == 0:
            raise ValueError("No dice available for reroll")
        
        # Roll new dice for available positions
        new_values = self._roll_dice(len(dice_state.available_dice))
        new_dice_state = dice_state.reroll_dice(new_values)
        
        # Update player state
        new_player_state = current_player.update_dice_state(new_dice_state)
        
        # Add to history
        history_event = {
            'type': 'reroll',
            'player': current_player.player_id,
            'old_dice': dice_state.values,
            'new_dice': new_dice_state.values,
            'reroll_count': new_dice_state.reroll_count
        }
        
        return game_state.update_player(
            current_player.player_id, new_player_state
        ).add_to_history(history_event)
    
    def _execute_keep_action(self, game_state: GameState, 
                            action_data: Optional[Dict]) -> GameState:
        """Execute keep dice action."""
        if not action_data or 'dice_indices' not in action_data:
            raise ValueError("Dice indices must be specified for keep action")
        
        current_player = game_state.current_player_state
        dice_state = current_player.dice_state
        
        dice_indices = action_data['dice_indices']
        if not all(0 <= i < len(dice_state.values) for i in dice_indices):
            raise ValueError("Invalid dice indices")
        
        # Keep specified dice
        new_dice_state = dice_state.keep_dice(dice_indices)
        
        # Update player state
        new_player_state = current_player.update_dice_state(new_dice_state)
        
        # Add to history
        history_event = {
            'type': 'keep',
            'player': current_player.player_id,
            'kept_indices': dice_indices,
            'kept_values': [dice_state.values[i] for i in dice_indices]
        }
        
        return game_state.update_player(
            current_player.player_id, new_player_state
        ).add_to_history(history_event)
    
    def get_game_statistics(self, game_state: GameState) -> Dict:
        """Get comprehensive game statistics."""
        stats = {
            'turn_number': game_state.turn_number,
            'max_turns': game_state.max_turns,
            'current_player': game_state.current_player,
            'phase': game_state.phase.value,
            'is_game_over': game_state.is_game_over,
            'players': []
        }
        
        for player in game_state.players:
            player_stats = {
                'player_id': player.player_id,
                'score': player.score,
                'turn_score': player.turn_score,
                'dice_values': player.dice_state.values,
                'reroll_count': player.dice_state.reroll_count,
                'actions_taken': player.actions_taken
            }
            stats['players'].append(player_stats)
        
        if game_state.is_game_over:
            stats['winner'] = game_state.winner
            stats['final_scores'] = [p.score for p in game_state.players]
        
        return stats
    
    def simulate_random_game(self, num_players: int, max_turns: int = 10) -> GameState:
        """Simulate a complete random game for testing."""
        game_state = self.create_initial_state(num_players, max_turns)
        
        while not game_state.is_game_over:
            valid_actions = self.get_valid_actions(game_state)
            if not valid_actions:
                break
            
            # Choose random action
            action = random.choice(valid_actions)
            
            # Execute action
            try:
                if action == DiceAction.SCORE:
                    game_state = self._execute_score_action(game_state)
                elif action == DiceAction.REROLL:
                    game_state = self._execute_reroll_action(game_state, {})
                elif action == DiceAction.KEEP:
                    # Randomly choose dice to keep
                    current_dice = game_state.current_player_state.dice_state
                    available_indices = [i for i in range(len(current_dice.values)) 
                                      if i not in current_dice.kept]
                    if available_indices:
                        keep_count = random.randint(1, len(available_indices))
                        keep_indices = random.sample(available_indices, keep_count)
                        game_state = self._execute_keep_action(game_state, 
                                                            {'dice_indices': keep_indices})
                    else:
                        # No dice to keep, try scoring
                        if DiceAction.SCORE in valid_actions:
                            game_state = self._execute_score_action(game_state)
                        else:
                            break
            except Exception:
                # If action fails, try to score or end turn
                break
        
        return game_state
