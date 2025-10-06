"""
Stress indicators for keystroke analysis.

This module aggregates extracted features into stress-relevant indicators
and provides methods for stress level classification.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from enum import Enum


class StressLevel(Enum):
    """Stress level classification."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3


class StressIndicators:
    """
    Aggregates keystroke features into stress indicators.
    
    This class provides methods to:
    - Calculate composite stress scores
    - Classify stress levels
    - Identify stress patterns
    - Generate stress reports
    """
    
    def __init__(self, thresholds: Dict[str, float] = None):
        """
        Initialize stress indicators.
        
        Args:
            thresholds: Custom thresholds for stress indicators
        """
        # Default thresholds based on research
        self.default_thresholds = {
            'high_error_rate': 0.15,      # >15% error rate indicates stress
            'high_pause_rate': 0.3,       # >30% pause rate indicates stress
            'high_variability': 0.8,      # >0.8 CV indicates stress
            'low_speed': 120,             # <120 CPM indicates stress
            'high_burst_rate': 0.4        # >40% burst rate indicates stress
        }
        
        self.thresholds = thresholds or self.default_thresholds
    
    def calculate_stress_score(self, features: Dict[str, float]) -> float:
        """
        Calculate a composite stress score from features.
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            Stress score between 0 (low stress) and 1 (high stress)
        """
        if not features:
            return 0.0
        
        # Normalize individual indicators
        indicators = []
        
        # Error rate indicator
        error_score = min(features.get('error_rate', 0) / self.thresholds['high_error_rate'], 1.0)
        indicators.append(error_score)
        
        # Pause rate indicator
        pause_score = min(features.get('pause_rate', 0) / self.thresholds['high_pause_rate'], 1.0)
        indicators.append(pause_score)
        
        # Variability indicator
        cv = features.get('interval_cv', 0)
        variability_score = min(cv / self.thresholds['high_variability'], 1.0)
        indicators.append(variability_score)
        
        # Speed indicator (inverted - lower speed = higher stress)
        cpm = features.get('cpm', 0)
        if cpm > 0:
            speed_score = max(0, 1 - (cpm / self.thresholds['low_speed']))
        else:
            speed_score = 1.0
        indicators.append(speed_score)
        
        # Burst rate indicator
        burst_score = min(features.get('burst_rate', 0) / self.thresholds['high_burst_rate'], 1.0)
        indicators.append(burst_score)
        
        # Calculate weighted average
        weights = [0.25, 0.25, 0.2, 0.2, 0.1]  # Error and pause rate weighted higher
        stress_score = np.average(indicators, weights=weights)
        
        return min(stress_score, 1.0)
    
    def classify_stress_level(self, features: Dict[str, float]) -> StressLevel:
        """
        Classify stress level based on features.
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            StressLevel enum value
        """
        stress_score = self.calculate_stress_score(features)
        
        if stress_score < 0.33:
            return StressLevel.LOW
        elif stress_score < 0.67:
            return StressLevel.MEDIUM
        else:
            return StressLevel.HIGH
    
    def get_stress_indicators(self, features: Dict[str, float]) -> Dict[str, Dict]:
        """
        Get detailed stress indicators for each feature category.
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            Dictionary of stress indicators by category
        """
        indicators = {
            'error_indicators': {
                'error_rate': features.get('error_rate', 0),
                'error_threshold': self.thresholds['high_error_rate'],
                'is_stressful': features.get('error_rate', 0) > self.thresholds['high_error_rate'],
                'stress_contribution': min(features.get('error_rate', 0) / self.thresholds['high_error_rate'], 1.0)
            },
            'pause_indicators': {
                'pause_rate': features.get('pause_rate', 0),
                'pause_threshold': self.thresholds['high_pause_rate'],
                'is_stressful': features.get('pause_rate', 0) > self.thresholds['high_pause_rate'],
                'stress_contribution': min(features.get('pause_rate', 0) / self.thresholds['high_pause_rate'], 1.0)
            },
            'variability_indicators': {
                'interval_cv': features.get('interval_cv', 0),
                'variability_threshold': self.thresholds['high_variability'],
                'is_stressful': features.get('interval_cv', 0) > self.thresholds['high_variability'],
                'stress_contribution': min(features.get('interval_cv', 0) / self.thresholds['high_variability'], 1.0)
            },
            'speed_indicators': {
                'cpm': features.get('cpm', 0),
                'speed_threshold': self.thresholds['low_speed'],
                'is_stressful': features.get('cpm', 0) < self.thresholds['low_speed'],
                'stress_contribution': max(0, 1 - (features.get('cpm', 0) / self.thresholds['low_speed']))
            },
            'burst_indicators': {
                'burst_rate': features.get('burst_rate', 0),
                'burst_threshold': self.thresholds['high_burst_rate'],
                'is_stressful': features.get('burst_rate', 0) > self.thresholds['high_burst_rate'],
                'stress_contribution': min(features.get('burst_rate', 0) / self.thresholds['high_burst_rate'], 1.0)
            }
        }
        
        return indicators
    
    def generate_stress_report(self, features: Dict[str, float], session_name: str = "Unknown") -> Dict:
        """
        Generate a comprehensive stress report.
        
        Args:
            features: Dictionary of extracted features
            session_name: Name of the typing session
            
        Returns:
            Dictionary containing stress report
        """
        stress_score = self.calculate_stress_score(features)
        stress_level = self.classify_stress_level(features)
        indicators = self.get_stress_indicators(features)
        
        # Count stressful indicators
        stressful_count = sum(1 for cat in indicators.values() if cat['is_stressful'])
        
        # Get top stress contributors
        contributors = []
        for category, data in indicators.items():
            if data['stress_contribution'] > 0.1:  # Only include significant contributors
                contributors.append({
                    'category': category.replace('_indicators', ''),
                    'contribution': data['stress_contribution'],
                    'value': data.get(list(data.keys())[0], 0)
                })
        
        # Sort by contribution
        contributors.sort(key=lambda x: x['contribution'], reverse=True)
        
        report = {
            'session_name': session_name,
            'stress_score': stress_score,
            'stress_level': stress_level.name,
            'stress_level_numeric': stress_level.value,
            'stressful_indicators_count': stressful_count,
            'top_contributors': contributors[:3],  # Top 3 contributors
            'indicators': indicators,
            'recommendations': self._generate_recommendations(stress_level, indicators)
        }
        
        return report
    
    def _generate_recommendations(self, stress_level: StressLevel, indicators: Dict) -> List[str]:
        """Generate recommendations based on stress level and indicators."""
        recommendations = []
        
        if stress_level == StressLevel.HIGH:
            recommendations.append("Consider taking a short break to reduce stress")
            recommendations.append("Try deep breathing exercises")
        
        if indicators['error_indicators']['is_stressful']:
            recommendations.append("High error rate detected - consider slowing down")
        
        if indicators['pause_indicators']['is_stressful']:
            recommendations.append("Frequent pauses detected - you may be distracted or fatigued")
        
        if indicators['speed_indicators']['is_stressful']:
            recommendations.append("Typing speed is below normal - consider if you're feeling stressed")
        
        if not recommendations:
            recommendations.append("Your typing patterns indicate normal stress levels")
        
        return recommendations
    
    def analyze_session_trends(self, session_features: List[Dict[str, float]], 
                             window_size: int = 10) -> Dict:
        """
        Analyze stress trends within a session.
        
        Args:
            session_features: List of feature dictionaries for different time windows
            window_size: Number of keystrokes per window
            
        Returns:
            Dictionary with trend analysis
        """
        if not session_features:
            return {}
        
        # Calculate stress scores for each window
        stress_scores = [self.calculate_stress_score(features) for features in session_features]
        
        # Calculate trends
        if len(stress_scores) > 1:
            trend_slope = np.polyfit(range(len(stress_scores)), stress_scores, 1)[0]
            trend_direction = "increasing" if trend_slope > 0.01 else "decreasing" if trend_slope < -0.01 else "stable"
        else:
            trend_slope = 0
            trend_direction = "stable"
        
        # Calculate statistics
        mean_stress = np.mean(stress_scores)
        max_stress = np.max(stress_scores)
        min_stress = np.min(stress_scores)
        stress_variance = np.var(stress_scores)
        
        return {
            'stress_scores': stress_scores,
            'mean_stress': mean_stress,
            'max_stress': max_stress,
            'min_stress': min_stress,
            'stress_variance': stress_variance,
            'trend_slope': trend_slope,
            'trend_direction': trend_direction,
            'window_count': len(session_features)
        }


def main():
    """Main function for command-line usage."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="KeyStress Stress Indicators")
    parser.add_argument("features_file", help="Path to features JSON file")
    parser.add_argument("--output", help="Output file for stress report (JSON)")
    
    args = parser.parse_args()
    
    # Load features
    with open(args.features_file, 'r') as f:
        features = json.load(f)
    
    # Analyze stress
    analyzer = StressIndicators()
    report = analyzer.generate_stress_report(features, "test_session")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Stress report saved to: {args.output}")
    else:
        print("Stress Report:")
        print(f"  Stress Score: {report['stress_score']:.3f}")
        print(f"  Stress Level: {report['stress_level']}")
        print(f"  Stressful Indicators: {report['stressful_indicators_count']}")
        print("\nTop Contributors:")
        for contrib in report['top_contributors']:
            print(f"  {contrib['category']}: {contrib['contribution']:.3f}")
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  - {rec}")


if __name__ == "__main__":
    main()
