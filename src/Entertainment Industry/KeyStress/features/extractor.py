"""
Feature extractor for keystroke data.

This module extracts stress-relevant features from keystroke logs including:
- Typing speed metrics
- Error rates and patterns
- Pause analysis
- Variability measures
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


class FeatureExtractor:
    """
    Extracts stress-relevant features from keystroke data.
    
    Features extracted:
    1. Typing Speed: CPM, average intervals, burst patterns
    2. Error Metrics: Backspace rate, correction patterns
    3. Pause Analysis: Pause frequency, duration distribution
    4. Variability: Standard deviation, coefficient of variation
    """
    
    def __init__(self, min_pause_threshold: float = 0.5):
        """
        Initialize the feature extractor.
        
        Args:
            min_pause_threshold: Minimum pause duration (seconds) to be considered a pause
        """
        self.min_pause_threshold = min_pause_threshold
        self.error_keys = {'backspace', 'delete'}
    
    def extract_features(self, keystrokes: List[Dict]) -> Dict[str, float]:
        """
        Extract all features from keystroke data.
        
        Args:
            keystrokes: List of keystroke dictionaries with 'key' and 'time' fields
            
        Returns:
            Dictionary of extracted features
        """
        if not keystrokes or len(keystrokes) < 2:
            return self._get_empty_features()
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(keystrokes)
        
        # Extract basic features
        features = {}
        
        # Typing speed features
        features.update(self._extract_speed_features(df))
        
        # Error features
        features.update(self._extract_error_features(df))
        
        # Pause features
        features.update(self._extract_pause_features(df))
        
        # Variability features
        features.update(self._extract_variability_features(df))
        
        # Burst features
        features.update(self._extract_burst_features(df))
        
        return features
    
    def _extract_speed_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract typing speed related features."""
        # Calculate inter-key intervals
        intervals = df['time'].diff().dropna()
        
        if len(intervals) == 0:
            return {
                'avg_interval': 0.0,
                'median_interval': 0.0,
                'cpm': 0.0,
                'wpm': 0.0
            }
        
        # Average and median intervals
        avg_interval = intervals.mean()
        median_interval = intervals.median()
        
        # Characters per minute (CPM)
        total_time = df['time'].iloc[-1] - df['time'].iloc[0]
        if total_time > 0:
            cpm = (len(df) / total_time) * 60
        else:
            cpm = 0.0
        
        # Words per minute (WPM) - rough estimate
        # Assuming average word length of 5 characters
        wpm = cpm / 5 if cpm > 0 else 0.0
        
        return {
            'avg_interval': avg_interval,
            'median_interval': median_interval,
            'cpm': cpm,
            'wpm': wpm
        }
    
    def _extract_error_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract error-related features."""
        total_keystrokes = len(df)
        if total_keystrokes == 0:
            return {
                'error_rate': 0.0,
                'backspace_rate': 0.0,
                'delete_rate': 0.0,
                'corrections_per_100': 0.0
            }
        
        # Count different types of errors
        backspace_count = len(df[df['key'] == 'backspace'])
        delete_count = len(df[df['key'] == 'delete'])
        total_errors = backspace_count + delete_count
        
        # Calculate rates
        error_rate = total_errors / total_keystrokes
        backspace_rate = backspace_count / total_keystrokes
        delete_rate = delete_count / total_keystrokes
        
        # Corrections per 100 characters
        corrections_per_100 = (total_errors / total_keystrokes) * 100
        
        return {
            'error_rate': error_rate,
            'backspace_rate': backspace_rate,
            'delete_rate': delete_rate,
            'corrections_per_100': corrections_per_100
        }
    
    def _extract_pause_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract pause-related features."""
        intervals = df['time'].diff().dropna()
        
        if len(intervals) == 0:
            return {
                'pause_count': 0.0,
                'pause_rate': 0.0,
                'avg_pause_duration': 0.0,
                'long_pause_rate': 0.0
            }
        
        # Identify pauses (intervals above threshold)
        pauses = intervals[intervals > self.min_pause_threshold]
        long_pauses = intervals[intervals > 2.0]  # Pauses longer than 2 seconds
        
        pause_count = len(pauses)
        pause_rate = pause_count / len(intervals)
        avg_pause_duration = pauses.mean() if len(pauses) > 0 else 0.0
        long_pause_rate = len(long_pauses) / len(intervals)
        
        return {
            'pause_count': pause_count,
            'pause_rate': pause_rate,
            'avg_pause_duration': avg_pause_duration,
            'long_pause_rate': long_pause_rate
        }
    
    def _extract_variability_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract variability-related features."""
        intervals = df['time'].diff().dropna()
        
        if len(intervals) < 2:
            return {
                'interval_std': 0.0,
                'interval_cv': 0.0,
                'interval_range': 0.0,
                'interval_iqr': 0.0
            }
        
        # Standard deviation
        interval_std = intervals.std()
        
        # Coefficient of variation
        interval_mean = intervals.mean()
        interval_cv = interval_std / interval_mean if interval_mean > 0 else 0.0
        
        # Range
        interval_range = intervals.max() - intervals.min()
        
        # Interquartile range
        interval_iqr = intervals.quantile(0.75) - intervals.quantile(0.25)
        
        return {
            'interval_std': interval_std,
            'interval_cv': interval_cv,
            'interval_range': interval_range,
            'interval_iqr': interval_iqr
        }
    
    def _extract_burst_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract burst typing features."""
        intervals = df['time'].diff().dropna()
        
        if len(intervals) == 0:
            return {
                'burst_count': 0.0,
                'avg_burst_length': 0.0,
                'burst_rate': 0.0
            }
        
        # Define burst as consecutive intervals < 0.1 seconds
        burst_threshold = 0.1
        is_burst = intervals < burst_threshold
        
        # Count bursts
        burst_count = 0
        current_burst_length = 0
        burst_lengths = []
        
        for is_burst_interval in is_burst:
            if is_burst_interval:
                current_burst_length += 1
            else:
                if current_burst_length > 0:
                    burst_count += 1
                    burst_lengths.append(current_burst_length)
                    current_burst_length = 0
        
        # Handle case where session ends with a burst
        if current_burst_length > 0:
            burst_count += 1
            burst_lengths.append(current_burst_length)
        
        avg_burst_length = np.mean(burst_lengths) if burst_lengths else 0.0
        burst_rate = burst_count / len(intervals) if len(intervals) > 0 else 0.0
        
        return {
            'burst_count': burst_count,
            'avg_burst_length': avg_burst_length,
            'burst_rate': burst_rate
        }
    
    def _get_empty_features(self) -> Dict[str, float]:
        """Return empty feature set when no data is available."""
        return {
            # Speed features
            'avg_interval': 0.0,
            'median_interval': 0.0,
            'cpm': 0.0,
            'wpm': 0.0,
            
            # Error features
            'error_rate': 0.0,
            'backspace_rate': 0.0,
            'delete_rate': 0.0,
            'corrections_per_100': 0.0,
            
            # Pause features
            'pause_count': 0.0,
            'pause_rate': 0.0,
            'avg_pause_duration': 0.0,
            'long_pause_rate': 0.0,
            
            # Variability features
            'interval_std': 0.0,
            'interval_cv': 0.0,
            'interval_range': 0.0,
            'interval_iqr': 0.0,
            
            # Burst features
            'burst_count': 0.0,
            'avg_burst_length': 0.0,
            'burst_rate': 0.0
        }
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        return list(self._get_empty_features().keys())
    
    def extract_features_from_file(self, log_file: str) -> Dict[str, float]:
        """
        Extract features from a keystroke log file.
        
        Args:
            log_file: Path to JSON log file
            
        Returns:
            Dictionary of extracted features
        """
        import json
        
        try:
            with open(log_file, 'r') as f:
                data = json.load(f)
            
            keystrokes = data.get('keystrokes', [])
            return self.extract_features(keystrokes)
        
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"Error reading log file {log_file}: {e}")
            return self._get_empty_features()


def main():
    """Main function for command-line usage."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="KeyStress Feature Extractor")
    parser.add_argument("log_file", help="Path to keystroke log file")
    parser.add_argument("--output", help="Output file for features (JSON)")
    
    args = parser.parse_args()
    
    extractor = FeatureExtractor()
    features = extractor.extract_features_from_file(args.log_file)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(features, f, indent=2)
        print(f"Features saved to: {args.output}")
    else:
        print("Extracted Features:")
        for feature, value in features.items():
            print(f"  {feature}: {value:.4f}")


if __name__ == "__main__":
    main()
