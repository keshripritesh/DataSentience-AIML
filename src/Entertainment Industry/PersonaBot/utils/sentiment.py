"""
Sentiment analysis utilities for PersonaBot
Analyzes user responses and provides feedback for personality adaptation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import logging

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Advanced sentiment analysis for conversational feedback"""
    
    def __init__(self):
        """Initialize sentiment analyzers"""
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Emotion keywords for enhanced analysis
        self.positive_emotions = {
            'joy': ['happy', 'excited', 'great', 'wonderful', 'amazing', 'fantastic'],
            'satisfaction': ['good', 'nice', 'fine', 'okay', 'alright', 'satisfied'],
            'enthusiasm': ['love', 'adore', 'enjoy', 'like', 'appreciate', 'thank'],
            'humor': ['funny', 'hilarious', 'lol', 'haha', '😄', '😂', '😊']
        }
        
        self.negative_emotions = {
            'anger': ['angry', 'mad', 'furious', 'annoyed', 'irritated', 'frustrated'],
            'sadness': ['sad', 'depressed', 'unhappy', 'disappointed', 'upset'],
            'disgust': ['disgusting', 'awful', 'terrible', 'horrible', 'bad'],
            'fear': ['scared', 'afraid', 'worried', 'anxious', 'nervous']
        }
        
        # Engagement indicators
        self.engagement_indicators = {
            'positive': ['yes', 'sure', 'absolutely', 'definitely', 'continue', 'more'],
            'negative': ['no', 'stop', 'enough', 'quit', 'end', 'bye', 'goodbye'],
            'questions': ['what', 'how', 'why', 'when', 'where', 'who', '?'],
            'continuation': ['and', 'also', 'moreover', 'furthermore', 'additionally']
        }
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Comprehensive sentiment analysis
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with sentiment scores and metrics
        """
        if not text or not text.strip():
            return self._empty_sentiment()
        
        text = text.lower().strip()
        
        # VADER sentiment analysis
        vader_scores = self.vader_analyzer.polarity_scores(text)
        
        # TextBlob sentiment analysis
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        # Emotion analysis
        emotion_scores = self._analyze_emotions(text)
        
        # Engagement analysis
        engagement_score = self._analyze_engagement(text)
        
        # Response quality indicators
        quality_indicators = self._analyze_response_quality(text)
        
        # Combine scores
        combined_sentiment = self._combine_sentiment_scores(
            vader_scores, textblob_polarity, emotion_scores, engagement_score
        )
        
        return {
            'vader_compound': vader_scores['compound'],
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'vader_neutral': vader_scores['neu'],
            'textblob_polarity': textblob_polarity,
            'textblob_subjectivity': textblob_subjectivity,
            'emotion_scores': emotion_scores,
            'engagement_score': engagement_score,
            'quality_indicators': quality_indicators,
            'combined_sentiment': combined_sentiment,
            'overall_score': combined_sentiment['overall']
        }
    
    def _analyze_emotions(self, text: str) -> Dict[str, float]:
        """Analyze emotional content of text"""
        emotion_scores = {}
        
        # Positive emotions
        for emotion, keywords in self.positive_emotions.items():
            score = sum(1 for keyword in keywords if keyword in text)
            emotion_scores[f'positive_{emotion}'] = min(score / len(keywords), 1.0)
        
        # Negative emotions
        for emotion, keywords in self.negative_emotions.items():
            score = sum(1 for keyword in keywords if keyword in text)
            emotion_scores[f'negative_{emotion}'] = min(score / len(keywords), 1.0)
        
        return emotion_scores
    
    def _analyze_engagement(self, text: str) -> float:
        """Analyze user engagement level"""
        words = text.split()
        if not words:
            return 0.0
        
        engagement_score = 0.0
        total_indicators = 0
        
        # Check for engagement indicators
        for category, indicators in self.engagement_indicators.items():
            matches = sum(1 for indicator in indicators if indicator in text)
            if category in ['positive', 'continuation']:
                engagement_score += matches
            elif category == 'negative':
                engagement_score -= matches
            elif category == 'questions':
                engagement_score += matches * 0.5  # Questions show interest
            total_indicators += len(indicators)
        
        # Normalize score
        if total_indicators > 0:
            engagement_score = engagement_score / total_indicators
        
        # Length factor (longer responses often indicate more engagement)
        length_factor = min(len(words) / 20.0, 1.0)  # Cap at 20 words
        engagement_score += length_factor * 0.2
        
        return np.clip(engagement_score, -1.0, 1.0)
    
    def _analyze_response_quality(self, text: str) -> Dict[str, float]:
        """Analyze response quality indicators"""
        words = text.split()
        
        return {
            'length': min(len(words) / 10.0, 1.0),  # Normalize by expected length
            'complexity': self._calculate_complexity(text),
            'specificity': self._calculate_specificity(text),
            'politeness': self._calculate_politeness(text)
        }
    
    def _calculate_complexity(self, text: str) -> float:
        """Calculate text complexity"""
        words = text.split()
        if not words:
            return 0.0
        
        # Average word length
        avg_word_length = np.mean([len(word) for word in words])
        
        # Vocabulary diversity (unique words / total words)
        unique_words = len(set(words))
        diversity = unique_words / len(words)
        
        # Sentence complexity (punctuation, capitalization)
        sentence_count = len(re.split(r'[.!?]+', text))
        complexity = (avg_word_length * 0.3 + diversity * 0.4 + sentence_count * 0.3) / 3
        
        return min(complexity, 1.0)
    
    def _calculate_specificity(self, text: str) -> float:
        """Calculate response specificity"""
        # Count specific terms (numbers, names, technical terms)
        specific_patterns = [
            r'\d+',  # Numbers
            r'[A-Z][a-z]+',  # Proper nouns
            r'\b(very|really|extremely|quite|somewhat)\b',  # Modifiers
            r'\b(this|that|these|those)\b'  # Demonstratives
        ]
        
        specificity_score = 0.0
        for pattern in specific_patterns:
            matches = len(re.findall(pattern, text))
            specificity_score += matches * 0.1
        
        return min(specificity_score, 1.0)
    
    def _calculate_politeness(self, text: str) -> float:
        """Calculate politeness level"""
        polite_terms = [
            'please', 'thank', 'thanks', 'sorry', 'excuse', 'pardon',
            'would you', 'could you', 'may i', 'kindly', 'appreciate'
        ]
        
        impolite_terms = [
            'shut up', 'stupid', 'idiot', 'dumb', 'fuck', 'shit',
            'damn', 'hell', 'ass', 'bitch'
        ]
        
        text_lower = text.lower()
        polite_count = sum(1 for term in polite_terms if term in text_lower)
        impolite_count = sum(1 for term in impolite_terms if term in text_lower)
        
        politeness = (polite_count - impolite_count * 2) / 10.0
        return np.clip(politeness, 0.0, 1.0)
    
    def _combine_sentiment_scores(self, 
                                 vader_scores: Dict[str, float],
                                 textblob_polarity: float,
                                 emotion_scores: Dict[str, float],
                                 engagement_score: float) -> Dict[str, float]:
        """Combine different sentiment scores into overall metrics"""
        
        # Weighted combination of sentiment scores
        vader_weight = 0.4
        textblob_weight = 0.3
        emotion_weight = 0.2
        engagement_weight = 0.1
        
        # Calculate emotion balance
        positive_emotions = sum(score for key, score in emotion_scores.items() 
                              if key.startswith('positive_'))
        negative_emotions = sum(score for key, score in emotion_scores.items() 
                              if key.startswith('negative_'))
        emotion_balance = (positive_emotions - negative_emotions) / max(positive_emotions + negative_emotions, 1)
        
        # Combined sentiment score
        overall_sentiment = (
            vader_scores['compound'] * vader_weight +
            textblob_polarity * textblob_weight +
            emotion_balance * emotion_weight +
            engagement_score * engagement_weight
        )
        
        return {
            'overall': np.clip(overall_sentiment, -1.0, 1.0),
            'confidence': self._calculate_confidence(vader_scores, textblob_polarity),
            'intensity': abs(overall_sentiment)
        }
    
    def _calculate_confidence(self, vader_scores: Dict[str, float], textblob_polarity: float) -> float:
        """Calculate confidence in sentiment analysis"""
        # Agreement between different methods
        vader_compound = vader_scores['compound']
        agreement = 1.0 - abs(vader_compound - textblob_polarity)
        
        # Strength of sentiment
        strength = max(abs(vader_compound), abs(textblob_polarity))
        
        # Neutrality penalty (neutral sentiment is less confident)
        neutrality_penalty = vader_scores['neu'] * 0.5
        
        confidence = (agreement * 0.4 + strength * 0.4 + (1.0 - neutrality_penalty) * 0.2)
        return np.clip(confidence, 0.0, 1.0)
    
    def _empty_sentiment(self) -> Dict[str, float]:
        """Return empty sentiment scores for empty text"""
        return {
            'vader_compound': 0.0,
            'vader_positive': 0.0,
            'vader_negative': 0.0,
            'vader_neutral': 1.0,
            'textblob_polarity': 0.0,
            'textblob_subjectivity': 0.0,
            'emotion_scores': {},
            'engagement_score': 0.0,
            'quality_indicators': {
                'length': 0.0,
                'complexity': 0.0,
                'specificity': 0.0,
                'politeness': 0.0
            },
            'combined_sentiment': {
                'overall': 0.0,
                'confidence': 0.0,
                'intensity': 0.0
            },
            'overall_score': 0.0
        }
    
    def extract_personality_feedback(self, sentiment_result: Dict[str, float]) -> Dict[str, float]:
        """
        Extract personality adaptation feedback from sentiment analysis
        
        Args:
            sentiment_result: Result from analyze_sentiment()
            
        Returns:
            Dictionary mapping personality traits to feedback scores
        """
        overall_score = sentiment_result['overall_score']
        emotion_scores = sentiment_result['emotion_scores']
        engagement_score = sentiment_result['engagement_score']
        quality_indicators = sentiment_result['quality_indicators']
        
        feedback = {}
        
        # Humor feedback
        humor_positive = emotion_scores.get('positive_humor', 0.0)
        feedback['humor'] = humor_positive * 0.5 + (overall_score + 1) * 0.25
        
        # Formality feedback
        politeness = quality_indicators['politeness']
        complexity = quality_indicators['complexity']
        feedback['formality'] = (politeness + complexity) / 2
        
        # Empathy feedback
        empathy_positive = emotion_scores.get('positive_joy', 0.0) + emotion_scores.get('positive_satisfaction', 0.0)
        empathy_negative = emotion_scores.get('negative_sadness', 0.0) + emotion_scores.get('negative_fear', 0.0)
        feedback['empathy'] = (empathy_positive - empathy_negative) * 0.5 + (overall_score + 1) * 0.25
        
        # Sarcasm feedback (negative sentiment might indicate sarcasm appreciation)
        sarcasm_positive = emotion_scores.get('positive_humor', 0.0)
        feedback['sarcasm'] = sarcasm_positive * 0.3 + (1 - politeness) * 0.2
        
        # Enthusiasm feedback
        enthusiasm_positive = emotion_scores.get('positive_enthusiasm', 0.0)
        feedback['enthusiasm'] = enthusiasm_positive * 0.6 + engagement_score * 0.4
        
        # Professionalism feedback
        feedback['professionalism'] = (politeness + complexity) / 2
        
        # Creativity feedback
        specificity = quality_indicators['specificity']
        feedback['creativity'] = specificity * 0.5 + (overall_score + 1) * 0.25
        
        # Assertiveness feedback
        feedback['assertiveness'] = (1 - politeness) * 0.3 + engagement_score * 0.4
        
        # Normalize all feedback to [-1, 1] range
        for trait in feedback:
            feedback[trait] = np.clip(feedback[trait] * 2 - 1, -1.0, 1.0)
        
        return feedback
