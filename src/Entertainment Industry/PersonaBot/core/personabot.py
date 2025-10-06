"""
Main PersonaBot class
Integrates all components for advanced conversational AI with personality adaptation
"""

import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging

from config.settings import settings
from core.nlp_engine import NLPEngine
from core.personality import PersonalityEncoder
from core.rl_agent import RLAgent
from core.reward import ConversationState, RewardFunction
from utils.sentiment import SentimentAnalyzer

logger = logging.getLogger(__name__)

class PersonaBot:
    """Advanced conversational AI with dynamic personality adaptation"""
    
    def __init__(self, 
                 initial_personality: Optional[Dict[str, float]] = None,
                 model_name: Optional[str] = None,
                 enable_rl: bool = True):
        """Initialize PersonaBot"""
        self.config = settings
        
        # Initialize components
        self.personality_encoder = PersonalityEncoder(initial_personality)
        self.nlp_engine = NLPEngine(model_name)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.reward_function = RewardFunction()
        
        # Initialize RL agent if enabled
        self.enable_rl = enable_rl
        if self.enable_rl:
            self.rl_agent = RLAgent(self.personality_encoder)
        else:
            self.rl_agent = None
        
        # Conversation state
        self.conversation_state = None
        self.conversation_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.performance_metrics = {
            'total_interactions': 0,
            'average_sentiment': 0.0,
            'average_engagement': 0.0,
            'personality_adaptations': 0,
            'session_start_time': datetime.now()
        }
        
        logger.info("PersonaBot initialized successfully")
    
    def start_conversation(self, context: str = "") -> str:
        """Start a new conversation"""
        # Initialize conversation state
        self.conversation_state = ConversationState(
            messages=[],
            personality_vector=self.personality_encoder.get_personality_vector(),
            conversation_length=0,
            start_time=datetime.now(),
            last_response_time=datetime.now(),
            engagement_metrics={
                'sentiment': 0.0,
                'engagement': 0.0,
                'relevance': 0.0,
                'coherence': 0.0
            }
        )
        
        # Generate welcome message
        welcome_message = self._generate_welcome_message(context)
        
        # Add to conversation history
        self.conversation_history.append({
            'role': 'assistant',
            'content': welcome_message,
            'timestamp': datetime.now(),
            'personality': self.personality_encoder.get_personality_dict()
        })
        
        logger.info("New conversation started")
        return welcome_message
    
    def chat(self, user_message: str) -> str:
        """Process user message and generate response"""
        if self.conversation_state is None:
            self.start_conversation()
        
        # Add user message to conversation state and history
        user_message_entry = {
            'role': 'user',
            'content': user_message,
            'timestamp': datetime.now(),
            'personality': self.personality_encoder.get_personality_dict()
        }
        self.conversation_state.messages.append(user_message_entry)
        self.conversation_history.append(user_message_entry)
        
        # Generate response
        try:
            response = self.nlp_engine.generate_response(
                user_message,
                self.conversation_state.personality_vector,
                self.nlp_engine.get_conversation_context()
            )
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            response = "I apologize, but I'm having trouble processing your message right now. Could you please try again?"
        
        # Add bot response to conversation state
        bot_message_entry = {
            'role': 'assistant',
            'content': response,
            'timestamp': datetime.now()
        }
        self.conversation_state.messages.append(bot_message_entry)
        
        # Update conversation state
        self.conversation_state.conversation_length += 1
        self.conversation_state.last_response_time = datetime.now()
        
        # Process interaction for RL learning
        detailed_rewards = {}
        if self.enable_rl and self.rl_agent:
            reward, detailed_rewards = self.rl_agent.process_interaction(
                user_message, response, self.conversation_state
            )
            
            # Update engagement metrics
            self.conversation_state.engagement_metrics.update({
                'sentiment': detailed_rewards.get('sentiment', 0.0),
                'engagement': detailed_rewards.get('engagement', 0.0),
                'relevance': detailed_rewards.get('relevance', 0.0),
                'coherence': detailed_rewards.get('coherence', 0.0)
            })
            
            # Update performance metrics
            self._update_performance_metrics(detailed_rewards)
        
        # Add bot response to conversation history
        bot_history_entry = {
            'role': 'assistant',
            'content': response,
            'timestamp': datetime.now(),
            'personality': self.personality_encoder.get_personality_dict(),
            'rewards': detailed_rewards if self.enable_rl else {}
        }
        self.conversation_history.append(bot_history_entry)
        
        return response
    
    def _generate_welcome_message(self, context: str = "") -> str:
        """Generate a personality-aware welcome message"""
        personality = self.personality_encoder.get_personality_dict()
        
        # Base welcome messages
        welcome_messages = {
            'formal': "Greetings! I'm here to assist you with any questions or topics you'd like to discuss.",
            'casual': "Hey there! I'm ready to chat about whatever's on your mind!",
            'enthusiastic': "Hello! I'm super excited to talk with you! What would you like to discuss?",
            'empathetic': "Hi! I'm here to listen and chat with you. How are you doing today?",
            'humorous': "Well hello there! Ready for some conversation? 😄",
            'professional': "Good day. I'm available to help you with any inquiries or discussions."
        }
        
        # Select welcome message based on dominant personality traits
        if personality['formality'] > 0.7:
            welcome = welcome_messages['formal']
        elif personality['enthusiasm'] > 0.7:
            welcome = welcome_messages['enthusiastic']
        elif personality['empathy'] > 0.7:
            welcome = welcome_messages['empathetic']
        elif personality['humor'] > 0.7:
            welcome = welcome_messages['humorous']
        elif personality['professionalism'] > 0.7:
            welcome = welcome_messages['professional']
        else:
            welcome = welcome_messages['casual']
        
        # Add context if provided
        if context:
            welcome += f" {context}"
        
        return welcome
    
    def _update_performance_metrics(self, detailed_rewards: Dict[str, float]) -> None:
        """Update performance tracking metrics"""
        self.performance_metrics['total_interactions'] += 1
        
        # Update averages
        total_interactions = self.performance_metrics['total_interactions']
        current_sentiment = detailed_rewards.get('sentiment', 0.0)
        current_engagement = detailed_rewards.get('engagement', 0.0)
        
        # Exponential moving average
        alpha = 0.1
        self.performance_metrics['average_sentiment'] = (
            alpha * current_sentiment + 
            (1 - alpha) * self.performance_metrics['average_sentiment']
        )
        self.performance_metrics['average_engagement'] = (
            alpha * current_engagement + 
            (1 - alpha) * self.performance_metrics['average_engagement']
        )
        
        # Count personality adaptations
        if any(abs(v) > 0.01 for v in detailed_rewards.values()):
            self.performance_metrics['personality_adaptations'] += 1
    
    def get_personality_summary(self) -> Dict[str, Any]:
        """Get current personality summary"""
        personality = self.personality_encoder.get_personality_dict()
        metrics = self.personality_encoder.get_personality_metrics()
        
        return {
            'current_traits': personality,
            'stability': metrics['stability'],
            'adaptation_count': metrics['adaptation_count'],
            'drift': metrics['drift'],
            'summary': self.personality_encoder.get_personality_summary()
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        session_duration = datetime.now() - self.performance_metrics['session_start_time']
        
        return {
            'total_interactions': self.performance_metrics['total_interactions'],
            'average_sentiment': self.performance_metrics['average_sentiment'],
            'average_engagement': self.performance_metrics['average_engagement'],
            'personality_adaptations': self.performance_metrics['personality_adaptations'],
            'session_duration': str(session_duration),
            'conversation_length': len(self.conversation_history),
            'rl_enabled': self.enable_rl,
            'rl_stats': self.rl_agent.get_policy_info() if self.rl_agent else None
        }
    
    def get_conversation_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get conversation history"""
        if limit is None:
            limit = settings.ui.max_display_messages
        
        return self.conversation_history[-limit:]
    
    def save_session(self, filepath: Optional[str] = None) -> str:
        """Save current session to file"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(settings.data.sessions_dir, f"session_{timestamp}.json")
        
        # Convert conversation history to serializable format
        serializable_history = []
        for item in self.conversation_history:
            serializable_item = {
                'role': item['role'],
                'content': item['content'],
                'timestamp': item['timestamp'].isoformat(),
                'personality': {k: float(v) for k, v in item['personality'].items()}
            }
            if 'rewards' in item:
                serializable_item['rewards'] = {k: float(v) for k, v in item['rewards'].items()}
            serializable_history.append(serializable_item)
        
        # Convert performance metrics to serializable format
        serializable_performance = {}
        for k, v in self.get_performance_summary().items():
            if isinstance(v, (np.integer, np.floating)):
                serializable_performance[k] = float(v)
            else:
                serializable_performance[k] = v
        
        # Convert personality summary to serializable format
        personality_summary = self.get_personality_summary()
        serializable_personality = {}
        for k, v in personality_summary.items():
            if k == 'current_traits':
                serializable_personality[k] = {trait: float(value) for trait, value in v.items()}
            elif isinstance(v, (np.integer, np.floating)):
                serializable_personality[k] = float(v)
            else:
                serializable_personality[k] = v
        
        session_data = {
            'conversation_history': serializable_history,
            'personality_summary': serializable_personality,
            'performance_summary': serializable_performance,
            'settings': self.config.to_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        logger.info(f"Session saved to {filepath}")
        return filepath
    
    def load_session(self, filepath: str) -> bool:
        """Load session from file"""
        if not os.path.exists(filepath):
            logger.error(f"Session file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'r') as f:
                session_data = json.load(f)
            
            # Load conversation history
            self.conversation_history = session_data.get('conversation_history', [])
            
            # Load personality state
            personality_summary = session_data.get('personality_summary', {})
            if 'current_traits' in personality_summary:
                # Reset personality to saved state
                self.personality_encoder = PersonalityEncoder(personality_summary['current_traits'])
            
            # Update conversation state if there's history
            if self.conversation_history:
                last_message = self.conversation_history[-1]
                self.conversation_state = ConversationState(
                    messages=[msg for msg in self.conversation_history if msg['role'] in ['user', 'assistant']],
                    personality_vector=self.personality_encoder.get_personality_vector(),
                    conversation_length=len(self.conversation_history),
                    start_time=datetime.fromisoformat(session_data['timestamp']),
                    last_response_time=datetime.now(),
                    engagement_metrics={
                        'sentiment': 0.0,
                        'engagement': 0.0,
                        'relevance': 0.0,
                        'coherence': 0.0
                    }
                )
            
            logger.info(f"Session loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading session: {e}")
            return False
    
    def reset_conversation(self) -> None:
        """Reset conversation state"""
        self.conversation_state = None
        self.conversation_history = []
        self.nlp_engine.reset_conversation()
        
        if self.rl_agent:
            self.rl_agent.reset()
        
        logger.info("Conversation reset")
    
    def set_personality(self, personality_traits: Dict[str, float]) -> None:
        """Set personality traits explicitly"""
        self.personality_encoder = PersonalityEncoder(personality_traits)
        
        if self.conversation_state:
            self.conversation_state.personality_vector = self.personality_encoder.get_personality_vector()
        
        logger.info(f"Personality set to: {personality_traits}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded models"""
        return {
            'nlp_engine': self.nlp_engine.get_model_info(),
            'personality_encoder': {
                'traits': settings.personality.traits,
                'current_values': self.personality_encoder.get_personality_dict()
            },
            'rl_agent': self.rl_agent.get_policy_info() if self.rl_agent else None
        }
    
    def export_personality(self, filepath: str) -> None:
        """Export current personality to file"""
        self.personality_encoder.save_personality(filepath)
    
    def import_personality(self, filepath: str) -> bool:
        """Import personality from file"""
        try:
            if not os.path.exists(filepath):
                logger.warning(f"Personality file not found: {filepath}")
                return False
                
            self.personality_encoder.load_personality(filepath)
            
            if self.conversation_state:
                self.conversation_state.personality_vector = self.personality_encoder.get_personality_vector()
            
            logger.info(f"Personality imported from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing personality: {e}")
            return False
    
    def get_available_sessions(self) -> List[str]:
        """Get list of available session files"""
        if not os.path.exists(settings.data.sessions_dir):
            return []
        
        session_files = []
        for filename in os.listdir(settings.data.sessions_dir):
            if filename.endswith('.json'):
                session_files.append(os.path.join(settings.data.sessions_dir, filename))
        
        return sorted(session_files, reverse=True)  # Most recent first
