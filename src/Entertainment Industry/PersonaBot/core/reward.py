"""
Reward functions for PersonaBot reinforcement learning
Defines how the bot learns from user interactions
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from utils.sentiment import SentimentAnalyzer
from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class ConversationState:
    """Represents the state of a conversation"""
    messages: List[Dict[str, str]]
    personality_vector: np.ndarray
    conversation_length: int
    start_time: datetime
    last_response_time: datetime
    engagement_metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "messages": self.messages,
            "personality_vector": self.personality_vector.tolist(),
            "conversation_length": self.conversation_length,
            "start_time": self.start_time.isoformat(),
            "last_response_time": self.last_response_time.isoformat(),
            "engagement_metrics": self.engagement_metrics
        }

class RewardFunction:
    """Advanced reward function for personality adaptation"""
    
    def __init__(self):
        """Initialize reward function"""
        self.sentiment_analyzer = SentimentAnalyzer()
        self.config = settings.reward
        
        # Conversation history for context
        self.conversation_history: List[ConversationState] = []
        
        # Performance tracking
        self.performance_metrics = {
            'total_rewards': 0.0,
            'average_reward': 0.0,
            'reward_count': 0,
            'conversation_count': 0
        }
    
    def calculate_reward(self, 
                        user_message: str,
                        bot_response: str,
                        conversation_state: ConversationState,
                        personality_vector: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """
        Calculate reward for the current interaction
        
        Args:
            user_message: User's input message
            bot_response: Bot's generated response
            conversation_state: Current conversation state
            personality_vector: Current personality vector
            
        Returns:
            Tuple of (total_reward, detailed_rewards)
        """
        # Analyze user sentiment
        sentiment_result = self.sentiment_analyzer.analyze_sentiment(user_message)
        
        # Calculate individual reward components
        engagement_reward = self._calculate_engagement_reward(
            user_message, bot_response, conversation_state, sentiment_result
        )
        
        sentiment_reward = self._calculate_sentiment_reward(sentiment_result)
        
        relevance_reward = self._calculate_relevance_reward(
            user_message, bot_response, conversation_state
        )
        
        coherence_reward = self._calculate_coherence_reward(
            bot_response, conversation_state
        )
        
        personality_reward = self._calculate_personality_reward(
            personality_vector, sentiment_result
        )
        
        # Combine rewards with weights
        total_reward = (
            engagement_reward * self.config.engagement_weight +
            sentiment_reward * self.config.sentiment_weight +
            relevance_reward * self.config.relevance_weight +
            coherence_reward * self.config.coherence_weight +
            personality_reward * 0.1  # Additional personality weight
        )
        
        detailed_rewards = {
            'engagement': engagement_reward,
            'sentiment': sentiment_reward,
            'relevance': relevance_reward,
            'coherence': coherence_reward,
            'personality': personality_reward,
            'total': total_reward
        }
        
        # Update performance metrics
        self._update_performance_metrics(total_reward)
        
        # Log reward calculation
        logger.debug(f"Reward calculated: {detailed_rewards}")
        
        return total_reward, detailed_rewards
    
    def _calculate_engagement_reward(self, 
                                   user_message: str,
                                   bot_response: str,
                                   conversation_state: ConversationState,
                                   sentiment_result: Dict[str, float]) -> float:
        """Calculate engagement-based reward"""
        engagement_score = sentiment_result['engagement_score']
        
        # Base engagement reward
        reward = engagement_score
        
        # Conversation length bonus (encourage longer conversations)
        length_bonus = min(conversation_state.conversation_length / 10.0, 1.0)
        reward += length_bonus * 0.2
        
        # Response time penalty (faster responses are better)
        response_time = datetime.now() - conversation_state.last_response_time
        time_penalty = min(response_time.total_seconds() / 30.0, 1.0)  # Penalty for slow responses
        reward -= time_penalty * 0.1
        
        # Question asking bonus (questions often indicate engagement)
        if '?' in user_message:
            reward += 0.1
        
        # Continuation indicators
        continuation_words = ['and', 'also', 'moreover', 'furthermore', 'additionally']
        if any(word in user_message.lower() for word in continuation_words):
            reward += 0.15
        
        return np.clip(reward, -1.0, 1.0)
    
    def _calculate_sentiment_reward(self, sentiment_result: Dict[str, float]) -> float:
        """Calculate sentiment-based reward"""
        overall_score = sentiment_result['overall_score']
        confidence = sentiment_result['combined_sentiment']['confidence']
        
        # Base sentiment reward
        reward = overall_score
        
        # Confidence bonus (more confident sentiment analysis gets higher reward)
        reward += confidence * 0.2
        
        # Emotion diversity bonus
        emotion_scores = sentiment_result['emotion_scores']
        emotion_count = len([score for score in emotion_scores.values() if score > 0.1])
        diversity_bonus = min(emotion_count / 5.0, 1.0) * 0.1
        reward += diversity_bonus
        
        return np.clip(reward, -1.0, 1.0)
    
    def _calculate_relevance_reward(self, 
                                  user_message: str,
                                  bot_response: str,
                                  conversation_state: ConversationState) -> float:
        """Calculate relevance-based reward"""
        # Simple keyword matching for relevance
        user_words = set(user_message.lower().split())
        bot_words = set(bot_response.lower().split())
        
        if not user_words:
            return 0.0
        
        # Word overlap
        overlap = len(user_words.intersection(bot_words))
        relevance_score = overlap / len(user_words)
        
        # Context relevance (check if response relates to conversation history)
        context_relevance = self._calculate_context_relevance(
            bot_response, conversation_state
        )
        
        # Topic consistency
        topic_consistency = self._calculate_topic_consistency(
            user_message, bot_response, conversation_state
        )
        
        total_relevance = (relevance_score * 0.4 + 
                          context_relevance * 0.3 + 
                          topic_consistency * 0.3)
        
        return np.clip(total_relevance, 0.0, 1.0)
    
    def _calculate_coherence_reward(self, 
                                  bot_response: str,
                                  conversation_state: ConversationState) -> float:
        """Calculate coherence-based reward"""
        # Response length appropriateness
        words = bot_response.split()
        length_score = 1.0
        
        if len(words) < self.config.min_response_length:
            length_score = len(words) / self.config.min_response_length
        elif len(words) > 50:  # Too long responses
            length_score = 50 / len(words)
        
        # Grammar and structure (simple heuristics)
        structure_score = self._calculate_structure_score(bot_response)
        
        # Personality consistency
        personality_consistency = self._calculate_personality_consistency(
            bot_response, conversation_state
        )
        
        total_coherence = (length_score * 0.3 + 
                          structure_score * 0.4 + 
                          personality_consistency * 0.3)
        
        return np.clip(total_coherence, 0.0, 1.0)
    
    def _calculate_personality_reward(self, 
                                    personality_vector: np.ndarray,
                                    sentiment_result: Dict[str, float]) -> float:
        """Calculate personality adaptation reward"""
        # Extract personality feedback from sentiment
        feedback = self.sentiment_analyzer.extract_personality_feedback(sentiment_result)
        
        # Calculate how well current personality matches user preferences
        personality_match = 0.0
        for trait, feedback_score in feedback.items():
            if trait in settings.personality.traits:
                trait_index = settings.personality.traits.index(trait)
                current_value = personality_vector[trait_index]
                
                # Reward for matching user preferences
                if feedback_score > 0 and current_value > 0.5:
                    personality_match += 0.1
                elif feedback_score < 0 and current_value < 0.5:
                    personality_match += 0.1
        
        # Stability bonus (avoid excessive personality changes)
        stability_bonus = 0.1  # Small bonus for maintaining personality
        
        return np.clip(personality_match + stability_bonus, -1.0, 1.0)
    
    def _calculate_context_relevance(self, 
                                   bot_response: str,
                                   conversation_state: ConversationState) -> float:
        """Calculate how relevant the response is to conversation context"""
        if len(conversation_state.messages) < 2:
            return 0.5  # Neutral score for new conversations
        
        # Check if response references previous messages
        recent_messages = conversation_state.messages[-4:]  # Last 4 messages
        context_words = set()
        
        for msg in recent_messages:
            context_words.update(msg.get('content', '').lower().split())
        
        response_words = set(bot_response.lower().split())
        
        if not context_words:
            return 0.5
        
        overlap = len(context_words.intersection(response_words))
        return min(overlap / len(context_words), 1.0)
    
    def _calculate_topic_consistency(self, 
                                   user_message: str,
                                   bot_response: str,
                                   conversation_state: ConversationState) -> float:
        """Calculate topic consistency between user message and bot response"""
        # Simple topic extraction (first few words)
        user_topic = ' '.join(user_message.split()[:3]).lower()
        bot_topic = ' '.join(bot_response.split()[:3]).lower()
        
        # Check for topic overlap
        user_words = set(user_topic.split())
        bot_words = set(bot_topic.split())
        
        if not user_words:
            return 0.5
        
        overlap = len(user_words.intersection(bot_words))
        return min(overlap / len(user_words), 1.0)
    
    def _calculate_structure_score(self, response: str) -> float:
        """Calculate structural quality of the response"""
        score = 0.5  # Base score
        
        # Check for proper sentence structure
        if response and response[0].isupper():
            score += 0.1
        
        if response and response[-1] in '.!?':
            score += 0.1
        
        # Check for reasonable word count
        words = response.split()
        if 3 <= len(words) <= 30:
            score += 0.2
        
        # Check for variety in vocabulary
        unique_words = len(set(words))
        if len(words) > 0:
            vocabulary_diversity = unique_words / len(words)
            score += vocabulary_diversity * 0.1
        
        return min(score, 1.0)
    
    def _calculate_personality_consistency(self, 
                                         response: str,
                                         conversation_state: ConversationState) -> float:
        """Calculate consistency with current personality"""
        # This is a simplified version - in practice, you'd use more sophisticated analysis
        personality_vector = conversation_state.personality_vector
        
        # Check for personality indicators in response
        consistency_score = 0.5  # Base score
        
        # Humor indicators
        humor_score = personality_vector[settings.personality.traits.index('humor')]
        humor_indicators = ['funny', 'hilarious', 'joke', 'lol', 'haha', '😄']
        if any(indicator in response.lower() for indicator in humor_indicators):
            consistency_score += humor_score * 0.1
        
        # Formality indicators
        formality_score = personality_vector[settings.personality.traits.index('formality')]
        formal_indicators = ['indeed', 'furthermore', 'moreover', 'consequently']
        informal_indicators = ['hey', 'cool', 'awesome', 'yeah']
        
        formal_count = sum(1 for indicator in formal_indicators if indicator in response.lower())
        informal_count = sum(1 for indicator in informal_indicators if indicator in response.lower())
        
        if formal_count > informal_count:
            consistency_score += formality_score * 0.1
        elif informal_count > formal_count:
            consistency_score += (1 - formality_score) * 0.1
        
        return min(consistency_score, 1.0)
    
    def _update_performance_metrics(self, reward: float) -> None:
        """Update performance tracking metrics"""
        self.performance_metrics['total_rewards'] += reward
        self.performance_metrics['reward_count'] += 1
        self.performance_metrics['average_reward'] = (
            self.performance_metrics['total_rewards'] / 
            self.performance_metrics['reward_count']
        )
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get performance summary"""
        return self.performance_metrics.copy()
    
    def reset_performance_metrics(self) -> None:
        """Reset performance metrics"""
        self.performance_metrics = {
            'total_rewards': 0.0,
            'average_reward': 0.0,
            'reward_count': 0,
            'conversation_count': 0
        }
