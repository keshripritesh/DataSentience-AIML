"""
Tests for personality module
Tests personality encoding, adaptation, and metrics
"""

import pytest
import numpy as np
import json
import tempfile
import os
from datetime import datetime
from unittest.mock import patch, MagicMock

from core.personality import PersonalityEncoder, PersonalityState
from config.settings import settings

class TestPersonalityState:
    """Test PersonalityState class"""
    
    def test_personality_state_creation(self):
        """Test creating a personality state"""
        traits = {"humor": 0.5, "formality": 0.7}
        state = PersonalityState(traits=traits, timestamp=datetime.now())
        
        assert state.traits == traits
        assert state.confidence == 1.0
        assert state.adaptation_rate == 0.1
    
    def test_to_vector(self):
        """Test converting personality state to vector"""
        traits = {
            "humor": 0.5, 
            "formality": 0.7, 
            "empathy": 0.3,
            "sarcasm": 0.2,
            "enthusiasm": 0.6,
            "professionalism": 0.8,
            "creativity": 0.4,
            "assertiveness": 0.5
        }
        state = PersonalityState(traits=traits, timestamp=datetime.now())
        
        vector = state.to_vector()
        expected = np.array([0.5, 0.7, 0.3, 0.2, 0.6, 0.8, 0.4, 0.5])
        
        np.testing.assert_array_almost_equal(vector, expected)
    
    def test_from_vector(self):
        """Test creating personality state from vector"""
        vector = np.array([0.5, 0.7, 0.3])
        state = PersonalityState.from_vector(vector, confidence=0.8)
        
        assert state.confidence == 0.8
        assert state.traits["humor"] == 0.5
        assert state.traits["formality"] == 0.7
        assert state.traits["empathy"] == 0.3
    
    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization"""
        traits = {"humor": 0.5, "formality": 0.7}
        original_state = PersonalityState(traits=traits, timestamp=datetime.now())
        
        # Convert to dict
        state_dict = original_state.to_dict()
        
        # Convert back from dict
        restored_state = PersonalityState.from_dict(state_dict)
        
        assert restored_state.traits == original_state.traits
        assert restored_state.confidence == original_state.confidence
        assert restored_state.adaptation_rate == original_state.adaptation_rate

class TestPersonalityEncoder:
    """Test PersonalityEncoder class"""
    
    def test_initialization_with_defaults(self):
        """Test initialization with default personality"""
        encoder = PersonalityEncoder()
        
        assert len(encoder.traits) == 8
        assert "humor" in encoder.traits
        assert "formality" in encoder.traits
        assert "empathy" in encoder.traits
    
    def test_initialization_with_custom_personality(self):
        """Test initialization with custom personality"""
        custom_personality = {
            "humor": 0.8, 
            "formality": 0.2,
            "empathy": 0.5,
            "sarcasm": 0.3,
            "enthusiasm": 0.6,
            "professionalism": 0.7,
            "creativity": 0.4,
            "assertiveness": 0.5
        }
        encoder = PersonalityEncoder(initial_personality=custom_personality)
        
        assert encoder.current_state.traits["humor"] == 0.8
        assert encoder.current_state.traits["formality"] == 0.2
    
    def test_get_personality_vector(self):
        """Test getting personality as vector"""
        custom_personality = {
            "humor": 0.5, 
            "formality": 0.7,
            "empathy": 0.3,
            "sarcasm": 0.2,
            "enthusiasm": 0.6,
            "professionalism": 0.8,
            "creativity": 0.4,
            "assertiveness": 0.5
        }
        encoder = PersonalityEncoder(initial_personality=custom_personality)
        
        vector = encoder.get_personality_vector()
        expected = np.array([0.5, 0.7, 0.3, 0.2, 0.6, 0.8, 0.4, 0.5])
        
        np.testing.assert_array_almost_equal(vector, expected)
    
    def test_get_personality_dict(self):
        """Test getting personality as dictionary"""
        custom_personality = {
            "humor": 0.5, 
            "formality": 0.7,
            "empathy": 0.3,
            "sarcasm": 0.2,
            "enthusiasm": 0.6,
            "professionalism": 0.8,
            "creativity": 0.4,
            "assertiveness": 0.5
        }
        encoder = PersonalityEncoder(initial_personality=custom_personality)
        
        personality_dict = encoder.get_personality_dict()
        
        assert personality_dict["humor"] == 0.5
        assert personality_dict["formality"] == 0.7
    
    def test_adapt_personality(self):
        """Test personality adaptation"""
        encoder = PersonalityEncoder()
        initial_traits = encoder.get_personality_dict()
        
        # Adapt personality
        feedback = {"humor": 0.5, "formality": -0.3}
        encoder.adapt_personality(feedback, learning_rate=0.1)
        
        new_traits = encoder.get_personality_dict()
        
        # Check that traits changed
        assert new_traits["humor"] != initial_traits["humor"]
        assert new_traits["formality"] != initial_traits["formality"]
        
        # Check that values are within bounds
        assert 0.0 <= new_traits["humor"] <= 1.0
        assert 0.0 <= new_traits["formality"] <= 1.0
    
    def test_adapt_personality_with_confidence(self):
        """Test personality adaptation with confidence"""
        encoder = PersonalityEncoder()
        initial_traits = encoder.get_personality_dict()
        
        # Adapt with low confidence
        feedback = {"humor": 0.5}
        encoder.adapt_personality(feedback, learning_rate=0.1, confidence=0.5)
        
        new_traits = encoder.get_personality_dict()
        
        # Change should be smaller with lower confidence
        change_with_low_confidence = abs(new_traits["humor"] - initial_traits["humor"])
        
        # Adapt with high confidence
        encoder.adapt_personality(feedback, learning_rate=0.1, confidence=1.0)
        
        new_traits_high_conf = encoder.get_personality_dict()
        change_with_high_confidence = abs(new_traits_high_conf["humor"] - new_traits["humor"])
        
        # High confidence should cause more change
        assert change_with_high_confidence > change_with_low_confidence
    
    def test_get_personality_prompt(self):
        """Test personality prompt generation"""
        custom_personality = {
            "humor": 0.8,
            "formality": 0.2,
            "empathy": 0.9,
            "sarcasm": 0.1,
            "enthusiasm": 0.7,
            "professionalism": 0.3,
            "creativity": 0.6,
            "assertiveness": 0.4
        }
        encoder = PersonalityEncoder(initial_personality=custom_personality)
        
        prompt = encoder.get_personality_prompt("Test context")
        
        assert "witty and humorous" in prompt
        assert "casual and friendly" in prompt
        assert "empathetic and understanding" in prompt
        assert "Test context" in prompt
    
    def test_get_personality_metrics(self):
        """Test personality metrics calculation"""
        encoder = PersonalityEncoder()
        
        # No history yet
        metrics = encoder.get_personality_metrics()
        assert metrics["stability"] == 1.0
        assert metrics["adaptation_count"] == 0
        assert metrics["drift"] == 0.0
        
        # Add some adaptations
        feedback = {"humor": 0.5}
        encoder.adapt_personality(feedback, learning_rate=0.1)
        
        metrics = encoder.get_personality_metrics()
        assert metrics["adaptation_count"] == 1
        assert metrics["drift"] > 0.0
    
    def test_save_and_load_personality(self):
        """Test saving and loading personality"""
        custom_personality = {
            "humor": 0.8, 
            "formality": 0.2,
            "empathy": 0.5,
            "sarcasm": 0.3,
            "enthusiasm": 0.6,
            "professionalism": 0.7,
            "creativity": 0.4,
            "assertiveness": 0.5
        }
        encoder = PersonalityEncoder(initial_personality=custom_personality)
        
        # Adapt personality
        feedback = {"humor": 0.3}
        encoder.adapt_personality(feedback, learning_rate=0.1)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            encoder.save_personality(filepath)
            
            # Create new encoder and load
            new_encoder = PersonalityEncoder()
            new_encoder.load_personality(filepath)
            
            # Check that personality was loaded correctly
            original_traits = encoder.get_personality_dict()
            loaded_traits = new_encoder.get_personality_dict()
            
            for trait in original_traits:
                assert abs(original_traits[trait] - loaded_traits[trait]) < 1e-6
            
            # Check history length
            assert len(new_encoder.history) == len(encoder.history)
            
        finally:
            os.unlink(filepath)
    
    def test_reset_personality(self):
        """Test resetting personality to defaults"""
        custom_personality = {
            "humor": 0.8, 
            "formality": 0.2,
            "empathy": 0.5,
            "sarcasm": 0.3,
            "enthusiasm": 0.6,
            "professionalism": 0.7,
            "creativity": 0.4,
            "assertiveness": 0.5
        }
        encoder = PersonalityEncoder(initial_personality=custom_personality)
        
        # Adapt personality
        feedback = {"humor": 0.3}
        encoder.adapt_personality(feedback, learning_rate=0.1)
        
        # Reset
        encoder.reset_personality()
        
        # Check that it's back to defaults
        default_traits = encoder.default_values
        current_traits = encoder.get_personality_dict()
        
        for trait in default_traits:
            assert abs(default_traits[trait] - current_traits[trait]) < 1e-6
        
        # Check that history is reset
        assert len(encoder.history) == 1
        assert len(encoder.adaptation_history) == 0
    
    def test_get_personality_summary(self):
        """Test personality summary generation"""
        custom_personality = {
            "humor": 0.8,
            "formality": 0.2,
            "empathy": 0.5
        }
        encoder = PersonalityEncoder(initial_personality=custom_personality)
        
        summary = encoder.get_personality_summary()
        
        assert "High humor" in summary
        assert "Low formality" in summary
        assert "Moderate empathy" in summary
    
    def test_personality_bounds(self):
        """Test that personality values stay within bounds"""
        encoder = PersonalityEncoder()
        
        # Try to set values outside bounds
        feedback = {"humor": 10.0, "formality": -5.0}
        encoder.adapt_personality(feedback, learning_rate=1.0)
        
        traits = encoder.get_personality_dict()
        
        # Values should be clamped to [0, 1]
        assert 0.0 <= traits["humor"] <= 1.0
        assert 0.0 <= traits["formality"] <= 1.0
    
    def test_empty_feedback(self):
        """Test adaptation with empty feedback"""
        encoder = PersonalityEncoder()
        initial_traits = encoder.get_personality_dict()
        
        # Adapt with empty feedback
        encoder.adapt_personality({}, learning_rate=0.1)
        
        new_traits = encoder.get_personality_dict()
        
        # Traits should remain unchanged
        for trait in initial_traits:
            assert abs(initial_traits[trait] - new_traits[trait]) < 1e-6
    
    def test_unknown_trait_feedback(self):
        """Test adaptation with unknown trait in feedback"""
        encoder = PersonalityEncoder()
        initial_traits = encoder.get_personality_dict()
        
        # Adapt with unknown trait
        feedback = {"unknown_trait": 0.5}
        encoder.adapt_personality(feedback, learning_rate=0.1)
        
        new_traits = encoder.get_personality_dict()
        
        # Known traits should remain unchanged
        for trait in initial_traits:
            assert abs(initial_traits[trait] - new_traits[trait]) < 1e-6
    
    def test_history_tracking(self):
        """Test that history is properly tracked"""
        encoder = PersonalityEncoder()
        
        # Initial state
        assert len(encoder.history) == 1
        assert len(encoder.adaptation_history) == 0
        
        # First adaptation
        feedback = {"humor": 0.5}
        encoder.adapt_personality(feedback, learning_rate=0.1)
        
        assert len(encoder.history) == 2
        assert len(encoder.adaptation_history) == 1
        
        # Second adaptation
        feedback = {"formality": 0.3}
        encoder.adapt_personality(feedback, learning_rate=0.1)
        
        assert len(encoder.history) == 3
        assert len(encoder.adaptation_history) == 2
    
    def test_adaptation_history_content(self):
        """Test that adaptation history contains correct information"""
        encoder = PersonalityEncoder()
        
        feedback = {"humor": 0.5, "formality": -0.3}
        encoder.adapt_personality(feedback, learning_rate=0.1)
        
        # Check adaptation history
        assert len(encoder.adaptation_history) == 1
        timestamp, adaptations = encoder.adaptation_history[0]
        
        assert isinstance(timestamp, datetime)
        assert "humor" in adaptations
        assert "formality" in adaptations
        
        # Check that adaptations reflect the feedback direction
        assert adaptations["humor"] > 0  # Positive feedback
        assert adaptations["formality"] < 0  # Negative feedback

if __name__ == "__main__":
    pytest.main([__file__])
