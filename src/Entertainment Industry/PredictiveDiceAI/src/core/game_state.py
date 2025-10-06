"""
Immutable game state representation for NeuralDicePredictor.

This module provides a comprehensive game state class that represents
the current state of a dice game, including dice values, scores,
turn information, and game history.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, FrozenSet
from enum import Enum
import numpy as np
from copy import deepcopy


class GamePhase(Enum):
    """Enumeration of game phases."""
    ROLLING = "rolling"
    SELECTING = "selecting"
    SCORING = "scoring"
    GAME_OVER = "game_over"


class DiceAction(Enum):
    """Enumeration of possible dice actions."""
    KEEP = "keep"
    REROLL = "reroll"
    SCORE = "score"


@dataclass(frozen=True)
class DiceState:
    """Immutable representation of dice state."""
    values: Tuple[int, ...]
    kept: FrozenSet[int] = field(default_factory=frozenset)
    reroll_count: int = 0
    
    def __post_init__(self):
        """Validate dice state."""
        if not all(1 <= v <= 6 for v in self.values):
            raise ValueError("Dice values must be between 1 and 6")
        if self.reroll_count < 0:
            raise ValueError("Reroll count cannot be negative")
        if len(self.kept) > len(self.values):
            raise ValueError("Cannot keep more dice than available")
    
    @property
    def available_dice(self) -> Tuple[int, ...]:
        """Get dice that can still be rerolled."""
        return tuple(v for i, v in enumerate(self.values) 
                    if i not in self.kept)
    
    @property
    def kept_values(self) -> Tuple[int, ...]:
        """Get values of kept dice."""
        return tuple(v for i, v in enumerate(self.values) 
                    if i in self.kept)
    
    def keep_dice(self, indices: List[int]) -> 'DiceState':
        """Create new state with specified dice kept."""
        if not all(0 <= i < len(self.values) for i in indices):
            raise ValueError("Invalid dice indices")
        new_kept = self.kept.union(indices)
        return DiceState(self.values, new_kept, self.reroll_count)
    
    def reroll_dice(self, new_values: List[int]) -> 'DiceState':
        """Create new state after rerolling."""
        if len(new_values) != len(self.available_dice):
            raise ValueError("New values count must match available dice")
        
        # Create new values array
        new_values_array = list(self.values)
        available_indices = [i for i in range(len(self.values)) 
                           if i not in self.kept]
        
        for i, new_val in zip(available_indices, new_values):
            new_values_array[i] = new_val
        
        return DiceState(
            tuple(new_values_array),
            self.kept,
            self.reroll_count + 1
        )


@dataclass(frozen=True)
class PlayerState:
    """Immutable representation of player state."""
    player_id: int
    score: int
    turn_score: int
    dice_state: DiceState
    actions_taken: Tuple[str, ...] = field(default_factory=tuple)
    
    def __post_init__(self):
        """Validate player state."""
        if self.score < 0:
            raise ValueError("Score cannot be negative")
        if self.turn_score < 0:
            raise ValueError("Turn score cannot be negative")
    
    def update_score(self, new_score: int) -> 'PlayerState':
        """Create new state with updated score."""
        return PlayerState(
            self.player_id,
            new_score,
            self.turn_score,
            self.dice_state,
            self.actions_taken
        )
    
    def update_turn_score(self, new_turn_score: int) -> 'PlayerState':
        """Create new state with updated turn score."""
        return PlayerState(
            self.player_id,
            self.score,
            new_turn_score,
            self.dice_state,
            self.actions_taken
        )
    
    def update_dice_state(self, new_dice_state: DiceState) -> 'PlayerState':
        """Create new state with updated dice state."""
        return PlayerState(
            self.player_id,
            self.score,
            self.turn_score,
            new_dice_state,
            self.actions_taken
        )
    
    def add_action(self, action: str) -> 'PlayerState':
        """Create new state with added action."""
        new_actions = self.actions_taken + (action,)
        return PlayerState(
            self.player_id,
            self.score,
            self.turn_score,
            self.dice_state,
            new_actions
        )


@dataclass(frozen=True)
class GameState:
    """Immutable representation of complete game state."""
    players: Tuple[PlayerState, ...]
    current_player: int
    phase: GamePhase
    turn_number: int
    max_turns: int
    game_rules: Dict[str, any] = field(default_factory=dict)
    game_history: Tuple[Dict[str, any], ...] = field(default_factory=tuple)
    
    def __post_init__(self):
        """Validate game state."""
        if not self.players:
            raise ValueError("Game must have at least one player")
        if not (0 <= self.current_player < len(self.players)):
            raise ValueError("Invalid current player index")
        if self.turn_number < 0:
            raise ValueError("Turn number cannot be negative")
        if self.max_turns < 1:
            raise ValueError("Max turns must be at least 1")
    
    @property
    def is_game_over(self) -> bool:
        """Check if game is over."""
        return (self.phase == GamePhase.GAME_OVER or 
                self.turn_number >= self.max_turns)
    
    @property
    def winner(self) -> Optional[int]:
        """Get the winner player ID if game is over."""
        if not self.is_game_over:
            return None
        
        max_score = max(p.score for p in self.players)
        winners = [p.player_id for p in self.players if p.score == max_score]
        return winners[0] if len(winners) == 1 else None
    
    @property
    def current_player_state(self) -> PlayerState:
        """Get current player's state."""
        return self.players[self.current_player]
    
    def next_turn(self) -> 'GameState':
        """Create new state for next turn."""
        next_player = (self.current_player + 1) % len(self.players)
        return GameState(
            self.players,
            next_player,
            GamePhase.ROLLING,
            self.turn_number + 1,
            self.max_turns,
            self.game_rules,
            self.game_history
        )
    
    def update_player(self, player_id: int, new_player_state: PlayerState) -> 'GameState':
        """Create new state with updated player."""
        if not (0 <= player_id < len(self.players)):
            raise ValueError("Invalid player ID")
        
        new_players = list(self.players)
        new_players[player_id] = new_player_state
        
        return GameState(
            tuple(new_players),
            self.current_player,
            self.phase,
            self.turn_number,
            self.max_turns,
            self.game_rules,
            self.game_history
        )
    
    def change_phase(self, new_phase: GamePhase) -> 'GameState':
        """Create new state with different phase."""
        return GameState(
            self.players,
            self.current_player,
            new_phase,
            self.turn_number,
            self.max_turns,
            self.game_rules,
            self.game_history
        )
    
    def add_to_history(self, event: Dict[str, any]) -> 'GameState':
        """Create new state with added history event."""
        new_history = self.game_history + (event,)
        return GameState(
            self.players,
            self.current_player,
            self.phase,
            self.turn_number,
            self.max_turns,
            self.game_rules,
            new_history
        )
    
    def to_tensor(self) -> np.ndarray:
        """Convert game state to neural network input tensor."""
        # This is a simplified tensor representation
        # In practice, you'd want a more sophisticated encoding
        
        # Player scores (normalized)
        max_possible_score = 1000  # Adjust based on game rules
        scores = np.array([p.score / max_possible_score for p in self.players])
        
        # Current dice values (one-hot encoded)
        dice_encoding = np.zeros((len(self.players), 6))
        for i, player in enumerate(self.players):
            for val in player.dice_state.values:
                dice_encoding[i, val - 1] += 1
        
        # Game phase (one-hot encoded)
        phase_encoding = np.zeros(len(GamePhase))
        phase_encoding[list(GamePhase).index(self.phase)] = 1
        
        # Turn information
        turn_info = np.array([
            self.current_player / len(self.players),
            self.turn_number / self.max_turns
        ])
        
        # Concatenate all features
        return np.concatenate([
            scores.flatten(),
            dice_encoding.flatten(),
            phase_encoding,
            turn_info
        ])
    
    def copy(self) -> 'GameState':
        """Create a deep copy of the game state."""
        return deepcopy(self)
