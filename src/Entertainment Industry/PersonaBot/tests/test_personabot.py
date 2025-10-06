"""
Tests for main PersonaBot class
Tests the integrated conversational AI system
"""

import pytest
import numpy as np
import json
import tempfile
import os
from datetime import datetime
from unittest.mock import patch, MagicMock

from core.personabot import PersonaBot
from core.reward import ConversationState
from config.settings import settings

class TestPersonaBot:
    """Test PersonaBot class"""
    
    def setup_method(self):
        """Setup method for each test"""
        self.initial_personality = {
            "humor": 0.5,
            "formality": 0.5,
            "empathy": 0.5,
            "sarcasm": 0.3,
            "enthusiasm": 0.6,
            "professionalism": 0.7,
            "creativity": 0.5,
            "assertiveness": 0.4
        }
    
    def test_initialization_with_defaults(self):
        """Test initialization with default settings"""
        bot = PersonaBot()
        
        assert bot.personality_encoder is not None
        assert bot.nlp_engine is not None
        assert bot.sentiment_analyzer is not None
        assert bot.reward_function is not None
        assert bot.enable_rl is True
        assert bot.rl_agent is not None
        assert bot.conversation_state is None
        assert len(bot.conversation_history) == 0
    
    def test_initialization_with_custom_personality(self):
        """Test initialization with custom personality"""
        bot = PersonaBot(initial_personality=self.initial_personality)
        
        personality = bot.personality_encoder.get_personality_dict()
        for trait, value in self.initial_personality.items():
            assert abs(personality[trait] - value) < 1e-6
    
    def test_initialization_without_rl(self):
        """Test initialization without reinforcement learning"""
        bot = PersonaBot(enable_rl=False)
        
        assert bot.enable_rl is False
        assert bot.rl_agent is None
    
    def test_start_conversation(self):
        """Test starting a conversation"""
        bot = PersonaBot(initial_personality=self.initial_personality)
        
        welcome_message = bot.start_conversation()
        
        assert isinstance(welcome_message, str)
        assert len(welcome_message) > 0
        assert bot.conversation_state is not None
        assert len(bot.conversation_history) == 1
        assert bot.conversation_history[0]['role'] == 'assistant'
        assert bot.conversation_history[0]['content'] == welcome_message
    
    def test_start_conversation_with_context(self):
        """Test starting a conversation with context"""
        bot = PersonaBot(initial_personality=self.initial_personality)
        
        context = "Let's talk about AI and machine learning."
        welcome_message = bot.start_conversation(context)
        
        assert context in welcome_message
    
    def test_chat_without_starting_conversation(self):
        """Test chat without explicitly starting conversation"""
        bot = PersonaBot(initial_personality=self.initial_personality)
        
        response = bot.chat("Hello!")
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert bot.conversation_state is not None
        assert len(bot.conversation_history) >= 2  # Welcome + response
    
    def test_chat_with_conversation_started(self):
        """Test chat with conversation already started"""
        bot = PersonaBot(initial_personality=self.initial_personality)
        bot.start_conversation()
        
        response = bot.chat("How are you?")
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert len(bot.conversation_history) >= 3  # Welcome + user + response
    
    def test_chat_updates_conversation_state(self):
        """Test that chat updates conversation state"""
        bot = PersonaBot(initial_personality=self.initial_personality)
        bot.start_conversation()
        
        initial_length = bot.conversation_state.conversation_length
        
        bot.chat("Test message")
        
        assert bot.conversation_state.conversation_length == initial_length + 1
    
    def test_chat_with_rl_enabled(self):
        """Test chat with reinforcement learning enabled"""
        bot = PersonaBot(initial_personality=self.initial_personality, enable_rl=True)
        bot.start_conversation()
        
        initial_personality = bot.personality_encoder.get_personality_dict()
        
        response = bot.chat("This is great!")
        
        # Check that RL agent processed the interaction
        assert bot.rl_agent is not None
        assert bot.rl_agent.training_stats['episodes'] > 0
    
    def test_chat_with_rl_disabled(self):
        """Test chat with reinforcement learning disabled"""
        bot = PersonaBot(initial_personality=self.initial_personality, enable_rl=False)
        bot.start_conversation()
        
        response = bot.chat("This is great!")
        
        # Check that RL agent is not used
        assert bot.rl_agent is None
    
    def test_get_personality_summary(self):
        """Test getting personality summary"""
        bot = PersonaBot(initial_personality=self.initial_personality)
        
        summary = bot.get_personality_summary()
        
        assert 'current_traits' in summary
        assert 'stability' in summary
        assert 'adaptation_count' in summary
        assert 'drift' in summary
        assert 'summary' in summary
        
        # Check that current traits match initial personality
        for trait, value in self.initial_personality.items():
            assert abs(summary['current_traits'][trait] - value) < 1e-6
    
    def test_get_performance_summary(self):
        """Test getting performance summary"""
        bot = PersonaBot(initial_personality=self.initial_personality)
        
        summary = bot.get_performance_summary()
        
        assert 'total_interactions' in summary
        assert 'average_sentiment' in summary
        assert 'average_engagement' in summary
        assert 'personality_adaptations' in summary
        assert 'session_duration' in summary
        assert 'conversation_length' in summary
        assert 'rl_enabled' in summary
        
        # Initial values should be zero
        assert summary['total_interactions'] == 0
        assert summary['conversation_length'] == 0
    
    def test_get_conversation_history(self):
        """Test getting conversation history"""
        bot = PersonaBot(initial_personality=self.initial_personality)
        bot.start_conversation()
        bot.chat("Hello!")
        bot.chat("How are you?")
        
        history = bot.get_conversation_history()
        
        assert len(history) >= 3  # Welcome + 2 exchanges
        
        # Check structure of history items
        for item in history:
            assert 'role' in item
            assert 'content' in item
            assert 'timestamp' in item
            assert item['role'] in ['user', 'assistant']
    
    def test_get_conversation_history_with_limit(self):
        """Test getting conversation history with limit"""
        bot = PersonaBot(initial_personality=self.initial_personality)
        bot.start_conversation()
        
        for i in range(5):
            bot.chat(f"Message {i}")
        
        # Get limited history
        history = bot.get_conversation_history(limit=3)
        
        assert len(history) <= 3
    
    def test_save_and_load_session(self):
        """Test saving and loading sessions"""
        bot = PersonaBot(initial_personality=self.initial_personality)
        bot.start_conversation()
        bot.chat("Hello!")
        bot.chat("How are you?")
        
        # Save session
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            saved_path = bot.save_session(filepath)
            
            # Create new bot and load session
            new_bot = PersonaBot()
            success = new_bot.load_session(saved_path)
            
            assert success is True
            assert len(new_bot.conversation_history) == len(bot.conversation_history)
            
            # Check that conversation state is restored
            assert new_bot.conversation_state is not None
            
        finally:
            os.unlink(filepath)
    
    def test_save_session_auto_filename(self):
        """Test saving session with auto-generated filename"""
        bot = PersonaBot(initial_personality=self.initial_personality)
        bot.start_conversation()
        bot.chat("Hello!")
        
        saved_path = bot.save_session()
        
        assert os.path.exists(saved_path)
        assert saved_path.endswith('.json')
        
        # Clean up
        os.unlink(saved_path)
    
    def test_load_nonexistent_session(self):
        """Test loading a session that doesn't exist"""
        bot = PersonaBot(initial_personality=self.initial_personality)
        
        success = bot.load_session("nonexistent_file.json")
        
        assert success is False
    
    def test_reset_conversation(self):
        """Test resetting conversation"""
        bot = PersonaBot(initial_personality=self.initial_personality)
        bot.start_conversation()
        bot.chat("Hello!")
        bot.chat("How are you?")
        
        # Reset conversation
        bot.reset_conversation()
        
        assert bot.conversation_state is None
        assert len(bot.conversation_history) == 0
    
    def test_reset_conversation_with_rl(self):
        """Test resetting conversation with RL agent"""
        bot = PersonaBot(initial_personality=self.initial_personality, enable_rl=True)
        bot.start_conversation()
        bot.chat("Hello!")
        
        # Reset conversation
        bot.reset_conversation()
        
        assert bot.conversation_state is None
        assert len(bot.conversation_history) == 0
        assert bot.rl_agent is not None  # RL agent should still exist
    
    def test_set_personality(self):
        """Test setting personality explicitly"""
        bot = PersonaBot(initial_personality=self.initial_personality)
        
        new_personality = {
            "humor": 0.8,
            "formality": 0.2,
            "empathy": 0.9,
            "sarcasm": 0.1,
            "enthusiasm": 0.7,
            "professionalism": 0.3,
            "creativity": 0.6,
            "assertiveness": 0.5
        }
        
        bot.set_personality(new_personality)
        
        current_personality = bot.personality_encoder.get_personality_dict()
        for trait, value in new_personality.items():
            assert abs(current_personality[trait] - value) < 1e-6
    
    def test_get_model_info(self):
        """Test getting model information"""
        bot = PersonaBot(initial_personality=self.initial_personality)
        
        model_info = bot.get_model_info()
        
        assert 'nlp_engine' in model_info
        assert 'personality_encoder' in model_info
        assert 'rl_agent' in model_info
        
        # Check NLP engine info
        nlp_info = model_info['nlp_engine']
        assert 'model_type' in nlp_info
        assert 'model_name' in nlp_info
        
        # Check personality encoder info
        personality_info = model_info['personality_encoder']
        assert 'traits' in personality_info
        assert 'current_values' in personality_info
        
        # Check RL agent info
        rl_info = model_info['rl_agent']
        assert rl_info is not None  # Should be present when RL is enabled
    
    def test_get_model_info_without_rl(self):
        """Test getting model information without RL"""
        bot = PersonaBot(initial_personality=self.initial_personality, enable_rl=False)
        
        model_info = bot.get_model_info()
        
        assert model_info['rl_agent'] is None
    
    def test_export_and_import_personality(self):
        """Test exporting and importing personality"""
        bot = PersonaBot(initial_personality=self.initial_personality)
        
        # Adapt personality
        bot.chat("This is great!")
        
        # Export personality
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            bot.export_personality(filepath)
            
            # Create new bot and import personality
            new_bot = PersonaBot()
            success = new_bot.import_personality(filepath)
            
            assert success is True
            
            # Check that personality was imported correctly
            original_personality = bot.personality_encoder.get_personality_dict()
            imported_personality = new_bot.personality_encoder.get_personality_dict()
            
            for trait in original_personality:
                assert abs(original_personality[trait] - imported_personality[trait]) < 1e-6
            
        finally:
            os.unlink(filepath)
    
    def test_import_nonexistent_personality(self):
        """Test importing personality from nonexistent file"""
        bot = PersonaBot(initial_personality=self.initial_personality)
        
        success = bot.import_personality("nonexistent_file.json")
        
        assert success is False
    
    def test_get_available_sessions(self):
        """Test getting available sessions"""
        bot = PersonaBot(initial_personality=self.initial_personality)
        
        # Create some test sessions with explicit filenames
        bot.start_conversation()
        bot.chat("Test message 1")
        session1_path = bot.save_session("data/sessions/test_session_1.json")
        
        # Add a small delay to ensure different timestamps
        import time
        time.sleep(0.1)
        
        bot.reset_conversation()
        bot.start_conversation()
        bot.chat("Test message 2")
        session2_path = bot.save_session("data/sessions/test_session_2.json")
        
        # Check that the session files were actually created
        assert os.path.exists(session1_path), f"Session 1 not created: {session1_path}"
        assert os.path.exists(session2_path), f"Session 2 not created: {session2_path}"
        
        available_sessions = bot.get_available_sessions()
        
        # Check that we can find at least our test sessions
        test_session_paths = [session1_path, session2_path]
        # Normalize paths for comparison
        normalized_available = [os.path.normpath(s) for s in available_sessions]
        normalized_test_paths = [os.path.normpath(s) for s in test_session_paths]
        found_test_sessions = [s for s in normalized_available if any(test_path in s for test_path in normalized_test_paths)]
        assert len(found_test_sessions) >= 2, f"Expected 2 sessions, found {len(found_test_sessions)}. Available: {available_sessions}"
        
        # Clean up test sessions
        for session_path in test_session_paths:
            if os.path.exists(session_path):
                os.unlink(session_path)
    
    def test_conversation_history_structure(self):
        """Test conversation history structure"""
        bot = PersonaBot(initial_personality=self.initial_personality)
        bot.start_conversation()
        bot.chat("Hello!")
        
        history = bot.get_conversation_history()
        
        for item in history:
            assert 'role' in item
            assert 'content' in item
            assert 'timestamp' in item
            assert 'personality' in item
            
            # Check for rewards only on assistant messages that are responses to user input
            # (not the initial welcome message)
            if bot.enable_rl and item['role'] == 'assistant' and 'rewards' in item:
                assert isinstance(item['rewards'], dict)
    
    def test_personality_adaptation_tracking(self):
        """Test that personality adaptations are tracked"""
        bot = PersonaBot(initial_personality=self.initial_personality, enable_rl=True)
        bot.start_conversation()
        
        initial_adaptations = bot.performance_metrics['personality_adaptations']
        
        bot.chat("This is amazing!")
        
        # Check that adaptations are tracked
        assert bot.performance_metrics['personality_adaptations'] >= initial_adaptations
    
    def test_performance_metrics_updates(self):
        """Test that performance metrics are updated"""
        bot = PersonaBot(initial_personality=self.initial_personality, enable_rl=True)
        bot.start_conversation()
        
        initial_interactions = bot.performance_metrics['total_interactions']
        
        bot.chat("Hello!")
        
        # Check that metrics are updated
        assert bot.performance_metrics['total_interactions'] == initial_interactions + 1
    
    def test_conversation_state_consistency(self):
        """Test conversation state consistency"""
        bot = PersonaBot(initial_personality=self.initial_personality)
        bot.start_conversation()
        
        # Check initial state
        assert bot.conversation_state is not None
        assert bot.conversation_state.conversation_length == 0
        
        bot.chat("First message")
        assert bot.conversation_state.conversation_length == 1
        
        bot.chat("Second message")
        assert bot.conversation_state.conversation_length == 2
    
    def test_welcome_message_personality_awareness(self):
        """Test that welcome message reflects personality"""
        # Test with high humor personality
        high_humor_personality = self.initial_personality.copy()
        high_humor_personality["humor"] = 0.9
        
        bot = PersonaBot(initial_personality=high_humor_personality)
        welcome_message = bot.start_conversation()
        
        # Should contain humor-related content
        assert any(word in welcome_message.lower() for word in ['funny', 'humorous', '😄', '😂'])
    
    def test_error_handling_in_chat(self):
        """Test error handling in chat method"""
        bot = PersonaBot(initial_personality=self.initial_personality)
        bot.start_conversation()
        
        # Mock NLP engine to raise an exception
        with patch.object(bot.nlp_engine, 'generate_response', side_effect=Exception("Test error")):
            response = bot.chat("This should cause an error")
            
            # Should still return a response (fallback)
            assert isinstance(response, str)
            assert len(response) > 0

if __name__ == "__main__":
    pytest.main([__file__])
