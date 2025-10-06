"""
Tests for feature extraction and stress indicators.

This module tests the feature extraction and stress analysis functionality.
"""

import pytest
import numpy as np
import pandas as pd
import json
import tempfile
import os
from unittest.mock import Mock, patch

from features.extractor import FeatureExtractor
from features.stress_indicators import StressIndicators, StressLevel


class TestFeatureExtractor:
    """Test cases for FeatureExtractor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = FeatureExtractor()
        
        # Sample keystroke data
        self.sample_keystrokes = [
            {"key": "h", "time": 0.0},
            {"key": "e", "time": 0.12},
            {"key": "l", "time": 0.25},
            {"key": "l", "time": 0.39},
            {"key": "o", "time": 0.52},
            {"key": "space", "time": 0.68},
            {"key": "w", "time": 0.85},
            {"key": "o", "time": 0.98},
            {"key": "r", "time": 1.12},
            {"key": "l", "time": 1.25},
            {"key": "d", "time": 1.38},
            {"key": "backspace", "time": 1.55},
            {"key": "d", "time": 1.68},
            {"key": "enter", "time": 1.82}
        ]
    
    def test_extract_features_basic(self):
        """Test basic feature extraction."""
        features = self.extractor.extract_features(self.sample_keystrokes)
        
        # Check that all expected features are present
        expected_features = [
            'avg_interval', 'median_interval', 'cpm', 'wpm',
            'error_rate', 'backspace_rate', 'delete_rate', 'corrections_per_100',
            'pause_count', 'pause_rate', 'avg_pause_duration', 'long_pause_rate',
            'interval_std', 'interval_cv', 'interval_range', 'interval_iqr',
            'burst_count', 'avg_burst_length', 'burst_rate'
        ]
        
        for feature in expected_features:
            assert feature in features
            assert isinstance(features[feature], (int, float))
    
    def test_extract_features_empty_data(self):
        """Test feature extraction with empty data."""
        features = self.extractor.extract_features([])
        
        # Should return empty features with zero values
        assert features['avg_interval'] == 0.0
        assert features['cpm'] == 0.0
        assert features['error_rate'] == 0.0
    
    def test_extract_features_single_keystroke(self):
        """Test feature extraction with single keystroke."""
        features = self.extractor.extract_features([{"key": "a", "time": 0.0}])
        
        # Should handle single keystroke gracefully
        assert features['avg_interval'] == 0.0
        assert features['cpm'] == 0.0
    
    def test_extract_speed_features(self):
        """Test speed feature extraction."""
        features = self.extractor._extract_speed_features(pd.DataFrame(self.sample_keystrokes))
        
        assert 'avg_interval' in features
        assert 'median_interval' in features
        assert 'cpm' in features
        assert 'wpm' in features
        
        # CPM should be positive for valid data
        assert features['cpm'] > 0
    
    def test_extract_error_features(self):
        """Test error feature extraction."""
        features = self.extractor._extract_error_features(pd.DataFrame(self.sample_keystrokes))
        
        assert 'error_rate' in features
        assert 'backspace_rate' in features
        assert 'delete_rate' in features
        assert 'corrections_per_100' in features
        
        # Should detect the backspace in sample data
        assert features['backspace_rate'] > 0
    
    def test_extract_pause_features(self):
        """Test pause feature extraction."""
        # Add some pauses to the data
        keystrokes_with_pauses = self.sample_keystrokes.copy()
        keystrokes_with_pauses.append({"key": "a", "time": 3.0})  # 1.18s pause
        
        features = self.extractor._extract_pause_features(pd.DataFrame(keystrokes_with_pauses))
        
        assert 'pause_count' in features
        assert 'pause_rate' in features
        assert 'avg_pause_duration' in features
        assert 'long_pause_rate' in features
    
    def test_extract_variability_features(self):
        """Test variability feature extraction."""
        features = self.extractor._extract_variability_features(pd.DataFrame(self.sample_keystrokes))
        
        assert 'interval_std' in features
        assert 'interval_cv' in features
        assert 'interval_range' in features
        assert 'interval_iqr' in features
    
    def test_extract_burst_features(self):
        """Test burst feature extraction."""
        features = self.extractor._extract_burst_features(pd.DataFrame(self.sample_keystrokes))
        
        assert 'burst_count' in features
        assert 'avg_burst_length' in features
        assert 'burst_rate' in features
    
    def test_extract_features_from_file(self):
        """Test feature extraction from file."""
        # Create temporary file with sample data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                "session_name": "test_session",
                "keystrokes": self.sample_keystrokes
            }, f)
            temp_file = f.name
        
        try:
            features = self.extractor.extract_features_from_file(temp_file)
            
            # Should extract features successfully
            assert 'avg_interval' in features
            assert 'cpm' in features
            assert 'error_rate' in features
            
        finally:
            os.unlink(temp_file)
    
    def test_extract_features_from_file_not_found(self):
        """Test feature extraction from non-existent file."""
        features = self.extractor.extract_features_from_file("nonexistent_file.json")
        
        # Should return empty features
        assert features['avg_interval'] == 0.0
        assert features['cpm'] == 0.0
    
    def test_get_feature_names(self):
        """Test getting feature names."""
        feature_names = self.extractor.get_feature_names()
        
        expected_features = [
            'avg_interval', 'median_interval', 'cpm', 'wpm',
            'error_rate', 'backspace_rate', 'delete_rate', 'corrections_per_100',
            'pause_count', 'pause_rate', 'avg_pause_duration', 'long_pause_rate',
            'interval_std', 'interval_cv', 'interval_range', 'interval_iqr',
            'burst_count', 'avg_burst_length', 'burst_rate'
        ]
        
        assert len(feature_names) == len(expected_features)
        for feature in expected_features:
            assert feature in feature_names


class TestStressIndicators:
    """Test cases for StressIndicators class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = StressIndicators()
        
        # Sample features
        self.sample_features = {
            'avg_interval': 0.2,
            'median_interval': 0.18,
            'cpm': 150,
            'wpm': 30,
            'error_rate': 0.1,
            'backspace_rate': 0.08,
            'delete_rate': 0.02,
            'corrections_per_100': 10,
            'pause_count': 5,
            'pause_rate': 0.25,
            'avg_pause_duration': 1.2,
            'long_pause_rate': 0.05,
            'interval_std': 0.15,
            'interval_cv': 0.75,
            'interval_range': 0.8,
            'interval_iqr': 0.2,
            'burst_count': 8,
            'avg_burst_length': 4,
            'burst_rate': 0.3
        }
    
    def test_calculate_stress_score(self):
        """Test stress score calculation."""
        stress_score = self.analyzer.calculate_stress_score(self.sample_features)
        
        # Stress score should be between 0 and 1
        assert 0 <= stress_score <= 1
        assert isinstance(stress_score, float)
    
    def test_calculate_stress_score_empty_features(self):
        """Test stress score calculation with empty features."""
        stress_score = self.analyzer.calculate_stress_score({})
        
        assert stress_score == 0.0
    
    def test_classify_stress_level(self):
        """Test stress level classification."""
        stress_level = self.analyzer.classify_stress_level(self.sample_features)
        
        assert isinstance(stress_level, StressLevel)
        assert stress_level in [StressLevel.LOW, StressLevel.MEDIUM, StressLevel.HIGH]
    
    def test_classify_stress_level_boundaries(self):
        """Test stress level classification boundaries."""
        # Test low stress
        low_stress_features = {
            'error_rate': 0.05,
            'pause_rate': 0.1,
            'interval_cv': 0.3,
            'cpm': 200,
            'burst_rate': 0.2
        }
        low_level = self.analyzer.classify_stress_level(low_stress_features)
        assert low_level == StressLevel.LOW
        
        # Test high stress
        high_stress_features = {
            'error_rate': 0.25,
            'pause_rate': 0.5,
            'interval_cv': 1.2,
            'cpm': 80,
            'burst_rate': 0.6
        }
        high_level = self.analyzer.classify_stress_level(high_stress_features)
        assert high_level == StressLevel.HIGH
    
    def test_get_stress_indicators(self):
        """Test getting stress indicators."""
        indicators = self.analyzer.get_stress_indicators(self.sample_features)
        
        expected_categories = [
            'error_indicators', 'pause_indicators', 'variability_indicators',
            'speed_indicators', 'burst_indicators'
        ]
        
        for category in expected_categories:
            assert category in indicators
            assert 'is_stressful' in indicators[category]
            assert 'stress_contribution' in indicators[category]
    
    def test_generate_stress_report(self):
        """Test stress report generation."""
        report = self.analyzer.generate_stress_report(self.sample_features, "test_session")
        
        expected_keys = [
            'session_name', 'stress_score', 'stress_level', 'stress_level_numeric',
            'stressful_indicators_count', 'top_contributors', 'indicators', 'recommendations'
        ]
        
        for key in expected_keys:
            assert key in report
        
        assert report['session_name'] == "test_session"
        assert 0 <= report['stress_score'] <= 1
        assert isinstance(report['stress_level'], str)
        assert isinstance(report['recommendations'], list)
    
    def test_analyze_session_trends(self):
        """Test session trend analysis."""
        # Create sample session features
        session_features = [
            {'error_rate': 0.05, 'pause_rate': 0.1, 'interval_cv': 0.3, 'cpm': 200, 'burst_rate': 0.2},
            {'error_rate': 0.1, 'pause_rate': 0.2, 'interval_cv': 0.5, 'cpm': 180, 'burst_rate': 0.25},
            {'error_rate': 0.15, 'pause_rate': 0.3, 'interval_cv': 0.7, 'cpm': 160, 'burst_rate': 0.3}
        ]
        
        trends = self.analyzer.analyze_session_trends(session_features)
        
        expected_keys = [
            'stress_scores', 'mean_stress', 'max_stress', 'min_stress',
            'stress_variance', 'trend_slope', 'trend_direction', 'window_count'
        ]
        
        for key in expected_keys:
            assert key in trends
        
        assert trends['window_count'] == 3
        assert len(trends['stress_scores']) == 3
    
    def test_custom_thresholds(self):
        """Test custom threshold configuration."""
        custom_thresholds = {
            'high_error_rate': 0.2,
            'high_pause_rate': 0.4,
            'high_variability': 1.0,
            'low_speed': 100,
            'high_burst_rate': 0.5
        }
        
        analyzer = StressIndicators(thresholds=custom_thresholds)
        
        # Test with features that should trigger different stress levels
        features = {
            'error_rate': 0.15,  # Below custom threshold
            'pause_rate': 0.3,   # Below custom threshold
            'interval_cv': 0.8,  # Below custom threshold
            'cpm': 120,          # Above custom threshold
            'burst_rate': 0.4    # Below custom threshold
        }
        
        stress_score = analyzer.calculate_stress_score(features)
        assert 0 <= stress_score <= 1


class TestFeatureExtractionIntegration:
    """Integration tests for feature extraction and stress analysis."""
    
    def test_end_to_end_analysis(self):
        """Test complete end-to-end feature extraction and stress analysis."""
        # Create sample keystroke data
        keystrokes = [
            {"key": "h", "time": 0.0},
            {"key": "e", "time": 0.15},
            {"key": "l", "time": 0.32},
            {"key": "l", "time": 0.48},
            {"key": "o", "time": 0.65},
            {"key": "backspace", "time": 0.85},
            {"key": "o", "time": 1.05},
            {"key": "space", "time": 1.25},
            {"key": "w", "time": 1.45},
            {"key": "o", "time": 1.68},
            {"key": "r", "time": 1.92},
            {"key": "l", "time": 2.15},
            {"key": "d", "time": 2.38},
            {"key": "enter", "time": 2.65}
        ]
        
        # Extract features
        extractor = FeatureExtractor()
        features = extractor.extract_features(keystrokes)
        
        # Analyze stress
        analyzer = StressIndicators()
        stress_score = analyzer.calculate_stress_score(features)
        stress_level = analyzer.classify_stress_level(features)
        report = analyzer.generate_stress_report(features, "test_session")
        
        # Verify results
        assert len(features) > 0
        assert 0 <= stress_score <= 1
        assert isinstance(stress_level, StressLevel)
        assert 'stress_score' in report
        assert 'recommendations' in report
    
    def test_feature_consistency(self):
        """Test that features are consistent across multiple extractions."""
        keystrokes = [
            {"key": "a", "time": 0.0},
            {"key": "b", "time": 0.1},
            {"key": "c", "time": 0.2},
            {"key": "d", "time": 0.3},
            {"key": "e", "time": 0.4}
        ]
        
        extractor = FeatureExtractor()
        
        # Extract features multiple times
        features1 = extractor.extract_features(keystrokes)
        features2 = extractor.extract_features(keystrokes)
        
        # Results should be identical
        for key in features1:
            assert features1[key] == features2[key]


if __name__ == "__main__":
    pytest.main([__file__])
