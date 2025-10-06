"""
Comprehensive tests for the game state module.

This module tests all aspects of the game state representation,
including dice state, player state, and game state classes.
"""

import pytest
import numpy as np
from unittest.mock import Mock

# Import the modules to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.game_state import (
    GamePhase, DiceAction, DiceState, PlayerState, GameState
)


class TestGamePhase:
    """Test the GamePhase enumeration."""
    
    def test_game_phase_values(self):
        """Test that GamePhase has the expected values."""
        assert GamePhase.ROLLING.value == "rolling"
        assert GamePhase.SELECTING.value == "selecting"
        assert GamePhase.SCORING.value == "scoring"
        assert GamePhase.GAME_OVER.value == "game_over"
    
    def test_game_phase_enumeration(self):
        """Test that all GamePhase values can be enumerated."""
        phases = list(GamePhase)
        assert len(phases) == 4
        assert all(isinstance(phase, GamePhase) for phase in phases)


class TestDiceAction:
    """Test the DiceAction enumeration."""
    
    def test_dice_action_values(self):
        """Test that DiceAction has the expected values."""
        assert DiceAction.KEEP.value == "keep"
        assert DiceAction.REROLL.value == "reroll"
        assert DiceAction.SCORE.value == "score"
    
    def test_dice_action_enumeration(self):
        """Test that all DiceAction values can be enumerated."""
        actions = list(DiceAction)
        assert len(actions) == 3
        assert all(isinstance(action, DiceAction) for action in actions)


class TestDiceState:
    """Test the DiceState class."""
    
    def test_dice_state_creation(self):
        """Test creating a DiceState with valid values."""
        dice_values = (1, 2, 3, 4, 5, 6)
        dice_state = DiceState(dice_values)
        
        assert dice_state.values == dice_values
        assert dice_state.kept == frozenset()
        assert dice_state.reroll_count == 0
    
    def test_dice_state_validation_valid_values(self):
        """Test that DiceState accepts valid dice values."""
        valid_values = [(1, 2, 3), (6, 6, 6), (1, 1, 1, 1, 1, 1)]
        
        for values in valid_values:
            dice_state = DiceState(values)
            assert dice_state.values == values
    
    def test_dice_state_validation_invalid_values(self):
        """Test that DiceState rejects invalid dice values."""
        invalid_values = [(0, 1, 2), (1, 2, 7), (1, 2, 3, 4, 5, 0)]
        
        for values in invalid_values:
            with pytest.raises(ValueError, match="Dice values must be between 1 and 6"):
                DiceState(values)
    
    def test_dice_state_validation_negative_reroll_count(self):
        """Test that DiceState rejects negative reroll count."""
        with pytest.raises(ValueError, match="Reroll count cannot be negative"):
            DiceState((1, 2, 3, 4, 5, 6), reroll_count=-1)
    
    def test_dice_state_validation_invalid_kept_count(self):
        """Test that DiceState rejects keeping more dice than available."""
        with pytest.raises(ValueError, match="Cannot keep more dice than available"):
            DiceState((1, 2, 3), kept=frozenset([0, 1, 2, 3]))
    
    def test_available_dice_property(self):
        """Test the available_dice property."""
        dice_state = DiceState((1, 2, 3, 4, 5, 6), kept=frozenset([0, 2, 4]))
        expected_available = (2, 4, 6)  # Indices 1, 3, 5
        assert dice_state.available_dice == expected_available
    
    def test_kept_values_property(self):
        """Test the kept_values property."""
        dice_state = DiceState((1, 2, 3, 4, 5, 6), kept=frozenset([0, 2, 4]))
        expected_kept = (1, 3, 5)  # Values at indices 0, 2, 4
        assert dice_state.kept_values == expected_kept
    
    def test_keep_dice_valid_indices(self):
        """Test keeping dice with valid indices."""
        dice_state = DiceState((1, 2, 3, 4, 5, 6))
        new_state = dice_state.keep_dice([0, 2, 4])
        
        assert new_state.kept == frozenset([0, 2, 4])
        assert new_state.values == dice_state.values
        assert new_state.reroll_count == dice_state.reroll_count
    
    def test_keep_dice_invalid_indices(self):
        """Test keeping dice with invalid indices."""
        dice_state = DiceState((1, 2, 3, 4, 5, 6))
        
        with pytest.raises(ValueError, match="Invalid dice indices"):
            dice_state.keep_dice([0, 6, 2])  # Index 6 is out of range
    
    def test_reroll_dice_valid_new_values(self):
        """Test rerolling dice with valid new values."""
        dice_state = DiceState((1, 2, 3, 4, 5, 6), kept=frozenset([0, 2, 4]))
        # Available dice are at indices 1, 3, 5 (values 2, 4, 6)
        new_values = [6, 6, 6]  # For the 3 available dice
        
        new_state = dice_state.reroll_dice(new_values)
        
        expected_values = (1, 6, 3, 6, 5, 6)  # Kept dice unchanged, others rerolled
        assert new_state.values == expected_values
        assert new_state.kept == dice_state.kept
        assert new_state.reroll_count == dice_state.reroll_count + 1
    
    def test_reroll_dice_invalid_new_values_count(self):
        """Test rerolling dice with wrong number of new values."""
        dice_state = DiceState((1, 2, 3, 4, 5, 6), kept=frozenset([0, 2, 4]))
        
        with pytest.raises(ValueError, match="New values count must match available dice"):
            dice_state.reroll_dice([6])  # Only 1 value for 2 available dice


class TestPlayerState:
    """Test the PlayerState class."""
    
    def test_player_state_creation(self):
        """Test creating a PlayerState with valid values."""
        dice_state = DiceState((1, 2, 3, 4, 5, 6))
        player_state = PlayerState(
            player_id=0,
            score=100,
            turn_score=50,
            dice_state=dice_state
        )
        
        assert player_state.player_id == 0
        assert player_state.score == 100
        assert player_state.turn_score == 50
        assert player_state.dice_state == dice_state
        assert player_state.actions_taken == ()
    
    def test_player_state_validation_negative_score(self):
        """Test that PlayerState rejects negative score."""
        dice_state = DiceState((1, 2, 3, 4, 5, 6))
        
        with pytest.raises(ValueError, match="Score cannot be negative"):
            PlayerState(0, -100, 50, dice_state)
    
    def test_player_state_validation_negative_turn_score(self):
        """Test that PlayerState rejects negative turn score."""
        dice_state = DiceState((1, 2, 3, 4, 5, 6))
        
        with pytest.raises(ValueError, match="Turn score cannot be negative"):
            PlayerState(0, 100, -50, dice_state)
    
    def test_update_score(self):
        """Test updating the player's score."""
        dice_state = DiceState((1, 2, 3, 4, 5, 6))
        player_state = PlayerState(0, 100, 50, dice_state)
        
        new_state = player_state.update_score(200)
        
        assert new_state.score == 200
        assert new_state.turn_score == 50  # Unchanged
        assert new_state.dice_state == dice_state  # Unchanged
        assert new_state.actions_taken == ()  # Unchanged
    
    def test_update_turn_score(self):
        """Test updating the player's turn score."""
        dice_state = DiceState((1, 2, 3, 4, 5, 6))
        player_state = PlayerState(0, 100, 50, dice_state)
        
        new_state = player_state.update_turn_score(75)
        
        assert new_state.score == 100  # Unchanged
        assert new_state.turn_score == 75
        assert new_state.dice_state == dice_state  # Unchanged
        assert new_state.actions_taken == ()  # Unchanged
    
    def test_update_dice_state(self):
        """Test updating the player's dice state."""
        old_dice_state = DiceState((1, 2, 3, 4, 5, 6))
        new_dice_state = DiceState((6, 6, 6, 4, 5, 6), kept=frozenset([0, 1, 2]))
        
        player_state = PlayerState(0, 100, 50, old_dice_state)
        new_state = player_state.update_dice_state(new_dice_state)
        
        assert new_state.score == 100  # Unchanged
        assert new_state.turn_score == 50  # Unchanged
        assert new_state.dice_state == new_dice_state
        assert new_state.actions_taken == ()  # Unchanged
    
    def test_add_action(self):
        """Test adding an action to the player's history."""
        dice_state = DiceState((1, 2, 3, 4, 5, 6))
        player_state = PlayerState(0, 100, 50, dice_state)
        
        new_state = player_state.add_action("score")
        
        assert new_state.score == 100  # Unchanged
        assert new_state.turn_score == 50  # Unchanged
        assert new_state.dice_state == dice_state  # Unchanged
        assert new_state.actions_taken == ("score",)
        
        # Add another action
        newer_state = new_state.add_action("reroll")
        assert newer_state.actions_taken == ("score", "reroll")


class TestGameState:
    """Test the GameState class."""
    
    def test_game_state_creation(self):
        """Test creating a GameState with valid values."""
        dice_state = DiceState((1, 2, 3, 4, 5, 6))
        player_state = PlayerState(0, 0, 0, dice_state)
        players = (player_state,)
        
        game_state = GameState(
            players=players,
            current_player=0,
            phase=GamePhase.ROLLING,
            turn_number=0,
            max_turns=10
        )
        
        assert game_state.players == players
        assert game_state.current_player == 0
        assert game_state.phase == GamePhase.ROLLING
        assert game_state.turn_number == 0
        assert game_state.max_turns == 10
        assert game_state.game_rules == {}
        assert game_state.game_history == ()
    
    def test_game_state_validation_no_players(self):
        """Test that GameState rejects empty players list."""
        with pytest.raises(ValueError, match="Game must have at least one player"):
            GameState(
                players=(),
                current_player=0,
                phase=GamePhase.ROLLING,
                turn_number=0,
                max_turns=10
            )
    
    def test_game_state_validation_invalid_current_player(self):
        """Test that GameState rejects invalid current player index."""
        dice_state = DiceState((1, 2, 3, 4, 5, 6))
        player_state = PlayerState(0, 0, 0, dice_state)
        players = (player_state,)
        
        with pytest.raises(ValueError, match="Invalid current player index"):
            GameState(
                players=players,
                current_player=1,  # Only 1 player (index 0)
                phase=GamePhase.ROLLING,
                turn_number=0,
                max_turns=10
            )
    
    def test_game_state_validation_negative_turn_number(self):
        """Test that GameState rejects negative turn number."""
        dice_state = DiceState((1, 2, 3, 4, 5, 6))
        player_state = PlayerState(0, 0, 0, dice_state)
        players = (player_state,)
        
        with pytest.raises(ValueError, match="Turn number cannot be negative"):
            GameState(
                players=players,
                current_player=0,
                phase=GamePhase.ROLLING,
                turn_number=-1,
                max_turns=10
            )
    
    def test_game_state_validation_invalid_max_turns(self):
        """Test that GameState rejects invalid max turns."""
        dice_state = DiceState((1, 2, 3, 4, 5, 6))
        player_state = PlayerState(0, 0, 0, dice_state)
        players = (player_state,)
        
        with pytest.raises(ValueError, match="Max turns must be at least 1"):
            GameState(
                players=players,
                current_player=0,
                phase=GamePhase.ROLLING,
                turn_number=0,
                max_turns=0
            )
    
    def test_is_game_over_property(self):
        """Test the is_game_over property."""
        dice_state = DiceState((1, 2, 3, 4, 5, 6))
        player_state = PlayerState(0, 0, 0, dice_state)
        players = (player_state,)
        
        # Game not over
        game_state = GameState(
            players=players,
            current_player=0,
            phase=GamePhase.ROLLING,
            turn_number=5,
            max_turns=10
        )
        assert not game_state.is_game_over
        
        # Game over by max turns
        game_state = GameState(
            players=players,
            current_player=0,
            phase=GamePhase.ROLLING,
            turn_number=10,
            max_turns=10
        )
        assert game_state.is_game_over
        
        # Game over by phase
        game_state = GameState(
            players=players,
            current_player=0,
            phase=GamePhase.GAME_OVER,
            turn_number=5,
            max_turns=10
        )
        assert game_state.is_game_over
    
    def test_winner_property(self):
        """Test the winner property."""
        # Game not over
        dice_state = DiceState((1, 2, 3, 4, 5, 6))
        player1 = PlayerState(0, 100, 0, dice_state)
        player2 = PlayerState(1, 200, 0, dice_state)
        players = (player1, player2)
        
        game_state = GameState(
            players=players,
            current_player=0,
            phase=GamePhase.ROLLING,
            turn_number=5,
            max_turns=10
        )
        assert game_state.winner is None
        
        # Game over with clear winner
        game_state = GameState(
            players=players,
            current_player=0,
            phase=GamePhase.GAME_OVER,
            turn_number=10,
            max_turns=10
        )
        assert game_state.winner == 1  # Player 2 has higher score
        
        # Game over with tie
        player1_tie = PlayerState(0, 200, 0, dice_state)
        player2_tie = PlayerState(1, 200, 0, dice_state)
        players_tie = (player1_tie, player2_tie)
        
        game_state_tie = GameState(
            players=players_tie,
            current_player=0,
            phase=GamePhase.GAME_OVER,
            turn_number=10,
            max_turns=10
        )
        assert game_state_tie.winner is None  # Tie game
    
    def test_current_player_state_property(self):
        """Test the current_player_state property."""
        dice_state = DiceState((1, 2, 3, 4, 5, 6))
        player1 = PlayerState(0, 100, 0, dice_state)
        player2 = PlayerState(1, 200, 0, dice_state)
        players = (player1, player2)
        
        game_state = GameState(
            players=players,
            current_player=1,
            phase=GamePhase.ROLLING,
            turn_number=0,
            max_turns=10
        )
        
        assert game_state.current_player_state == player2
    
    def test_next_turn(self):
        """Test the next_turn method."""
        dice_state = DiceState((1, 2, 3, 4, 5, 6))
        player1 = PlayerState(0, 100, 0, dice_state)
        player2 = PlayerState(1, 200, 0, dice_state)
        players = (player1, player2)
        
        game_state = GameState(
            players=players,
            current_player=0,
            phase=GamePhase.ROLLING,
            turn_number=5,
            max_turns=10
        )
        
        new_state = game_state.next_turn()
        
        assert new_state.current_player == 1  # Next player
        assert new_state.phase == GamePhase.ROLLING  # Reset to rolling
        assert new_state.turn_number == 6  # Incremented
        assert new_state.players == players  # Unchanged
        assert new_state.max_turns == 10  # Unchanged
        
        # Test wrapping around to first player
        new_state = new_state.next_turn()
        assert new_state.current_player == 0
    
    def test_update_player(self):
        """Test the update_player method."""
        dice_state = DiceState((1, 2, 3, 4, 5, 6))
        player1 = PlayerState(0, 100, 0, dice_state)
        player2 = PlayerState(1, 200, 0, dice_state)
        players = (player1, player2)
        
        game_state = GameState(
            players=players,
            current_player=0,
            phase=GamePhase.ROLLING,
            turn_number=5,
            max_turns=10
        )
        
        # Update player 1
        updated_player1 = PlayerState(0, 150, 0, dice_state)
        new_state = game_state.update_player(0, updated_player1)
        
        assert new_state.players[0] == updated_player1
        assert new_state.players[1] == player2  # Unchanged
        assert new_state.current_player == 0  # Unchanged
        assert new_state.phase == GamePhase.ROLLING  # Unchanged
    
    def test_update_player_invalid_id(self):
        """Test updating player with invalid ID."""
        dice_state = DiceState((1, 2, 3, 4, 5, 6))
        player1 = PlayerState(0, 100, 0, dice_state)
        players = (player1,)
        
        game_state = GameState(
            players=players,
            current_player=0,
            phase=GamePhase.ROLLING,
            turn_number=5,
            max_turns=10
        )
        
        updated_player = PlayerState(1, 150, 0, dice_state)
        
        with pytest.raises(ValueError, match="Invalid player ID"):
            game_state.update_player(1, updated_player)
    
    def test_change_phase(self):
        """Test the change_phase method."""
        dice_state = DiceState((1, 2, 3, 4, 5, 6))
        player_state = PlayerState(0, 100, 0, dice_state)
        players = (player_state,)
        
        game_state = GameState(
            players=players,
            current_player=0,
            phase=GamePhase.ROLLING,
            turn_number=5,
            max_turns=10
        )
        
        new_state = game_state.change_phase(GamePhase.SCORING)
        
        assert new_state.phase == GamePhase.SCORING
        assert new_state.players == players  # Unchanged
        assert new_state.current_player == 0  # Unchanged
        assert new_state.turn_number == 5  # Unchanged
    
    def test_add_to_history(self):
        """Test the add_to_history method."""
        dice_state = DiceState((1, 2, 3, 4, 5, 6))
        player_state = PlayerState(0, 100, 0, dice_state)
        players = (player_state,)
        
        game_state = GameState(
            players=players,
            current_player=0,
            phase=GamePhase.ROLLING,
            turn_number=5,
            max_turns=10
        )
        
        event = {'type': 'score', 'points': 100}
        new_state = game_state.add_to_history(event)
        
        assert len(new_state.game_history) == 1
        assert new_state.game_history[0] == event
        assert new_state.players == players  # Unchanged
        assert new_state.phase == GamePhase.ROLLING  # Unchanged
    
    def test_to_tensor(self):
        """Test the to_tensor method."""
        dice_state = DiceState((1, 2, 3, 4, 5, 6))
        player1 = PlayerState(0, 100, 0, dice_state)
        player2 = PlayerState(1, 200, 0, dice_state)
        players = (player1, player2)
        
        game_state = GameState(
            players=players,
            current_player=0,
            phase=GamePhase.ROLLING,
            turn_number=5,
            max_turns=10
        )
        
        tensor = game_state.to_tensor()
        
        # Check that tensor is numpy array
        assert isinstance(tensor, np.ndarray)
        
        # Check tensor shape (should be 1D)
        assert tensor.ndim == 1
        
        # Check that tensor contains expected data
        assert tensor.size > 0
    
    def test_copy(self):
        """Test the copy method."""
        dice_state = DiceState((1, 2, 3, 4, 5, 6))
        player_state = PlayerState(0, 100, 0, dice_state)
        players = (player_state,)
        
        game_state = GameState(
            players=players,
            current_player=0,
            phase=GamePhase.ROLLING,
            turn_number=5,
            max_turns=10
        )
        
        copied_state = game_state.copy()
        
        # Check that it's a copy, not the same object
        assert copied_state is not game_state
        
        # Check that all attributes are copied
        assert copied_state.players == game_state.players
        assert copied_state.current_player == game_state.current_player
        assert copied_state.phase == game_state.phase
        assert copied_state.turn_number == game_state.turn_number
        assert copied_state.max_turns == game_state.max_turns


if __name__ == "__main__":
    pytest.main([__file__])
