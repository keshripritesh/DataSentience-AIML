"""
Tests for sentiment analysis module
Tests sentiment analysis, emotion detection, and personality feedback extraction
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from utils.sentiment import SentimentAnalyzer

class TestSentimentAnalyzer:
    """Test SentimentAnalyzer class"""
    
    def setup_method(self):
        """Setup method for each test"""
        self.analyzer = SentimentAnalyzer()
    
    def test_initialization(self):
        """Test analyzer initialization"""
        assert self.analyzer.vader_analyzer is not None
        assert len(self.analyzer.positive_emotions) > 0
        assert len(self.analyzer.negative_emotions) > 0
        assert len(self.analyzer.engagement_indicators) > 0
    
    def test_analyze_sentiment_empty_text(self):
        """Test sentiment analysis with empty text"""
        result = self.analyzer.analyze_sentiment("")
        
        assert result['vader_compound'] == 0.0
        assert result['vader_neutral'] == 1.0
        assert result['overall_score'] == 0.0
        assert result['engagement_score'] == 0.0
    
    def test_analyze_sentiment_positive_text(self):
        """Test sentiment analysis with positive text"""
        text = "I love this! It's amazing and wonderful!"
        result = self.analyzer.analyze_sentiment(text)
        
        assert result['vader_compound'] > 0.0
        assert result['vader_positive'] > 0.0
        assert result['overall_score'] > 0.0
        assert result['engagement_score'] > 0.0
    
    def test_analyze_sentiment_negative_text(self):
        """Test sentiment analysis with negative text"""
        text = "I hate this! It's terrible and awful!"
        result = self.analyzer.analyze_sentiment(text)
        
        assert result['vader_compound'] < 0.0
        assert result['vader_negative'] > 0.0
        assert result['overall_score'] < 0.0
    
    def test_analyze_sentiment_neutral_text(self):
        """Test sentiment analysis with neutral text"""
        text = "This is a neutral statement about something."
        result = self.analyzer.analyze_sentiment(text)
        
        assert abs(result['vader_compound']) < 0.1
        assert result['vader_neutral'] > 0.5
        assert abs(result['overall_score']) < 0.1
    
    def test_analyze_emotions_positive(self):
        """Test emotion analysis with positive emotions"""
        text = "I'm so happy and excited about this amazing news!"
        emotions = self.analyzer._analyze_emotions(text)
        
        assert emotions.get('positive_joy', 0.0) > 0.0
        # Check for any positive emotion being detected
        positive_emotions = [v for k, v in emotions.items() if k.startswith('positive_') and v > 0.0]
        assert len(positive_emotions) > 0
    
    def test_analyze_emotions_negative(self):
        """Test emotion analysis with negative emotions"""
        text = "I'm so sad and worried about this terrible situation."
        emotions = self.analyzer._analyze_emotions(text)
        
        assert emotions.get('negative_sadness', 0.0) > 0.0
        assert emotions.get('negative_fear', 0.0) > 0.0
    
    def test_analyze_emotions_humor(self):
        """Test emotion analysis with humor"""
        text = "That's so funny! Haha, this is hilarious!"
        emotions = self.analyzer._analyze_emotions(text)
        
        assert emotions.get('positive_humor', 0.0) > 0.0
    
    def test_analyze_engagement_positive(self):
        """Test engagement analysis with positive indicators"""
        text = "Yes, absolutely! I want to continue and learn more."
        engagement = self.analyzer._analyze_engagement(text)
        
        assert engagement > 0.0
    
    def test_analyze_engagement_negative(self):
        """Test engagement analysis with negative indicators"""
        text = "No, stop. I'm done. Goodbye."
        engagement = self.analyzer._analyze_engagement(text)
        
        assert engagement < 0.0
    
    def test_analyze_engagement_questions(self):
        """Test engagement analysis with questions"""
        text = "What is this? How does it work? Why is it like that?"
        engagement = self.analyzer._analyze_engagement(text)
        
        assert engagement > 0.0
    
    def test_analyze_engagement_length(self):
        """Test engagement analysis with different text lengths"""
        short_text = "Ok"
        long_text = "This is a much longer response that should indicate higher engagement because it contains more words and shows more interest in the conversation."
        
        short_engagement = self.analyzer._analyze_engagement(short_text)
        long_engagement = self.analyzer._analyze_engagement(long_text)
        
        assert long_engagement > short_engagement
    
    def test_analyze_response_quality(self):
        """Test response quality analysis"""
        text = "This is a very specific response with numbers 123 and proper nouns like John."
        quality = self.analyzer._analyze_response_quality(text)
        
        assert 'length' in quality
        assert 'complexity' in quality
        assert 'specificity' in quality
        assert 'politeness' in quality
        
        assert 0.0 <= quality['length'] <= 1.0
        assert 0.0 <= quality['complexity'] <= 1.0
        assert 0.0 <= quality['specificity'] <= 1.0
        assert 0.0 <= quality['politeness'] <= 1.0
    
    def test_calculate_complexity(self):
        """Test complexity calculation"""
        simple_text = "This is simple."
        complex_text = "This is a more sophisticated and intricate response with diverse vocabulary."
        
        simple_complexity = self.analyzer._calculate_complexity(simple_text)
        complex_complexity = self.analyzer._calculate_complexity(complex_text)
        
        assert complex_complexity > simple_complexity
    
    def test_calculate_specificity(self):
        """Test specificity calculation"""
        vague_text = "Something happened somewhere."
        specific_text = "John Smith visited the Eiffel Tower in Paris on March 15th, 2023."
        
        vague_specificity = self.analyzer._calculate_specificity(vague_text)
        specific_specificity = self.analyzer._calculate_specificity(specific_text)
        
        assert specific_specificity > vague_specificity
    
    def test_calculate_politeness(self):
        """Test politeness calculation"""
        polite_text = "Please, could you kindly help me? Thank you very much."
        impolite_text = "Shut up, you idiot! This is stupid!"
        
        polite_score = self.analyzer._calculate_politeness(polite_text)
        impolite_score = self.analyzer._calculate_politeness(impolite_text)
        
        assert polite_score > impolite_score
    
    def test_combine_sentiment_scores(self):
        """Test sentiment score combination"""
        vader_scores = {'compound': 0.5, 'pos': 0.6, 'neg': 0.1, 'neu': 0.3}
        textblob_polarity = 0.4
        emotion_scores = {'positive_joy': 0.3, 'negative_sadness': 0.1}
        engagement_score = 0.2
        
        combined = self.analyzer._combine_sentiment_scores(
            vader_scores, textblob_polarity, emotion_scores, engagement_score
        )
        
        assert 'overall' in combined
        assert 'confidence' in combined
        assert 'intensity' in combined
        
        assert -1.0 <= combined['overall'] <= 1.0
        assert 0.0 <= combined['confidence'] <= 1.0
        assert 0.0 <= combined['intensity'] <= 1.0
    
    def test_calculate_confidence(self):
        """Test confidence calculation"""
        # High agreement between methods
        vader_scores = {'compound': 0.5, 'pos': 0.6, 'neg': 0.1, 'neu': 0.3}
        textblob_polarity = 0.5
        
        high_confidence = self.analyzer._calculate_confidence(vader_scores, textblob_polarity)
        
        # Low agreement between methods
        vader_scores = {'compound': 0.5, 'pos': 0.6, 'neg': 0.1, 'neu': 0.3}
        textblob_polarity = -0.5
        
        low_confidence = self.analyzer._calculate_confidence(vader_scores, textblob_polarity)
        
        assert high_confidence > low_confidence
    
    def test_extract_personality_feedback_positive(self):
        """Test personality feedback extraction with positive sentiment"""
        sentiment_result = {
            'overall_score': 0.8,
            'emotion_scores': {
                'positive_humor': 0.7,
                'positive_joy': 0.6,
                'positive_enthusiasm': 0.5
            },
            'engagement_score': 0.6,
            'quality_indicators': {
                'politeness': 0.8,
                'complexity': 0.7,
                'specificity': 0.6
            }
        }
        
        feedback = self.analyzer.extract_personality_feedback(sentiment_result)
        
        assert 'humor' in feedback
        assert 'formality' in feedback
        assert 'empathy' in feedback
        assert 'enthusiasm' in feedback
        
        # Positive sentiment should lead to positive feedback for most traits
        assert feedback['humor'] > 0.0
        assert feedback['enthusiasm'] > 0.0
    
    def test_extract_personality_feedback_negative(self):
        """Test personality feedback extraction with negative sentiment"""
        sentiment_result = {
            'overall_score': -0.6,
            'emotion_scores': {
                'negative_sadness': 0.7,
                'negative_anger': 0.5
            },
            'engagement_score': -0.3,
            'quality_indicators': {
                'politeness': 0.2,
                'complexity': 0.4,
                'specificity': 0.3
            }
        }
        
        feedback = self.analyzer.extract_personality_feedback(sentiment_result)
        
        # Negative sentiment should lead to negative feedback for most traits
        assert feedback['empathy'] < 0.0
    
    def test_extract_personality_feedback_humor(self):
        """Test personality feedback extraction with humor"""
        sentiment_result = {
            'overall_score': 0.5,
            'emotion_scores': {
                'positive_humor': 0.9,
                'positive_joy': 0.3
            },
            'engagement_score': 0.4,
            'quality_indicators': {
                'politeness': 0.6,
                'complexity': 0.5,
                'specificity': 0.4
            }
        }
        
        feedback = self.analyzer.extract_personality_feedback(sentiment_result)
        
        # High humor should lead to positive humor feedback
        assert feedback['humor'] > 0.0
    
    def test_extract_personality_feedback_formality(self):
        """Test personality feedback extraction with formality indicators"""
        sentiment_result = {
            'overall_score': 0.3,
            'emotion_scores': {},
            'engagement_score': 0.4,
            'quality_indicators': {
                'politeness': 0.9,  # High politeness
                'complexity': 0.8,  # High complexity
                'specificity': 0.6
            }
        }
        
        feedback = self.analyzer.extract_personality_feedback(sentiment_result)
        
        # High politeness and complexity should lead to positive formality feedback
        assert feedback['formality'] > 0.0
        assert feedback['professionalism'] > 0.0
    
    def test_extract_personality_feedback_empathy(self):
        """Test personality feedback extraction with empathy indicators"""
        sentiment_result = {
            'overall_score': 0.4,
            'emotion_scores': {
                'positive_joy': 0.7,
                'positive_satisfaction': 0.6,
                'negative_sadness': 0.1,
                'negative_fear': 0.1
            },
            'engagement_score': 0.5,
            'quality_indicators': {
                'politeness': 0.7,
                'complexity': 0.6,
                'specificity': 0.5
            }
        }
        
        feedback = self.analyzer.extract_personality_feedback(sentiment_result)
        
        # Positive emotions should lead to positive empathy feedback
        assert feedback['empathy'] > 0.0
    
    def test_extract_personality_feedback_bounds(self):
        """Test that personality feedback is within bounds"""
        sentiment_result = {
            'overall_score': 1.0,  # Maximum positive
            'emotion_scores': {
                'positive_humor': 1.0,
                'positive_joy': 1.0,
                'positive_enthusiasm': 1.0
            },
            'engagement_score': 1.0,
            'quality_indicators': {
                'politeness': 1.0,
                'complexity': 1.0,
                'specificity': 1.0
            }
        }
        
        feedback = self.analyzer.extract_personality_feedback(sentiment_result)
        
        # All feedback should be within [-1, 1] bounds
        for trait, value in feedback.items():
            assert -1.0 <= value <= 1.0
    
    def test_analyze_sentiment_with_emojis(self):
        """Test sentiment analysis with emojis"""
        text_with_emojis = "I'm so happy! 😄 This is amazing! 😊"
        result = self.analyzer.analyze_sentiment(text_with_emojis)
        
        # Should detect positive emotions including humor from emojis
        assert result['emotion_scores'].get('positive_humor', 0.0) > 0.0
        assert result['overall_score'] > 0.0
    
    def test_analyze_sentiment_with_questions(self):
        """Test sentiment analysis with questions"""
        text_with_questions = "What do you think? How does this work? Can you explain?"
        result = self.analyzer.analyze_sentiment(text_with_questions)
        
        # Questions should increase engagement
        assert result['engagement_score'] > 0.0
    
    def test_analyze_sentiment_mixed_emotions(self):
        """Test sentiment analysis with mixed emotions"""
        text = "I'm happy but also a bit worried about the future."
        result = self.analyzer.analyze_sentiment(text)
        
        # Should detect both positive and negative emotions
        assert result['emotion_scores'].get('positive_joy', 0.0) > 0.0
        assert result['emotion_scores'].get('negative_fear', 0.0) > 0.0
    
    def test_analyze_sentiment_very_long_text(self):
        """Test sentiment analysis with very long text"""
        long_text = "This is a very long text that contains many words and should test the robustness of the sentiment analysis system. " * 10
        result = self.analyzer.analyze_sentiment(long_text)
        
        # Should handle long text without errors
        assert 'overall_score' in result
        assert 'engagement_score' in result
        assert 'quality_indicators' in result
    
    def test_analyze_sentiment_special_characters(self):
        """Test sentiment analysis with special characters"""
        text_with_special = "This text has special chars: @#$%^&*() and numbers 12345!"
        result = self.analyzer.analyze_sentiment(text_with_special)
        
        # Should handle special characters without errors
        assert 'overall_score' in result
        assert 'quality_indicators' in result

if __name__ == "__main__":
    pytest.main([__file__])
