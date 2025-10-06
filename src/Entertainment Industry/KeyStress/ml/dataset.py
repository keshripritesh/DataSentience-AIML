"""
Dataset preparation for KeyStress.

This module handles the collection and formatting of keystroke logs
into training samples for stress detection models.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from features.extractor import FeatureExtractor
from capture.session_recorder import SessionRecorder


class Dataset:
    """
    Dataset preparation for stress detection from keystroke data.
    
    This class handles:
    - Loading keystroke logs
    - Feature extraction
    - Label preparation
    - Train/test splitting
    - Data preprocessing
    """
    
    def __init__(self, data_dir: str = "data", logs_dir: str = "data/logs"):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing session metadata
            logs_dir: Directory containing keystroke logs
        """
        self.data_dir = data_dir
        self.logs_dir = logs_dir
        self.feature_extractor = FeatureExtractor()
        self.session_recorder = SessionRecorder(data_dir)
        
        # Data storage
        self.features_df = None
        self.labels = None
        self.feature_names = None
        
        # Preprocessing
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_data(self, min_sessions: int = 5, include_unlabeled: bool = False) -> bool:
        """
        Load and prepare the dataset from keystroke logs.
        
        Args:
            min_sessions: Minimum number of sessions required
            include_unlabeled: Whether to include sessions without stress labels
            
        Returns:
            True if data was loaded successfully
        """
        # Get all sessions
        sessions = self.session_recorder.list_sessions(completed_only=True)
        
        if len(sessions) < min_sessions:
            print(f"Not enough sessions. Found {len(sessions)}, need at least {min_sessions}")
            return False
        
        # Extract features and labels
        features_list = []
        labels_list = []
        session_names = []
        
        for session in sessions:
            # Skip sessions without stress labels if not including unlabeled
            if not include_unlabeled and session.stress_level is None:
                continue
            
            # Load keystroke data
            log_file = os.path.join(self.logs_dir, f"{session.session_name}.json")
            if not os.path.exists(log_file):
                print(f"Log file not found: {log_file}")
                continue
            
            try:
                # Extract features
                features = self.feature_extractor.extract_features_from_file(log_file)
                
                if features:
                    features_list.append(features)
                    labels_list.append(session.stress_level or 2)  # Default to medium stress
                    session_names.append(session.session_name)
            
            except Exception as e:
                print(f"Error processing session {session.session_name}: {e}")
                continue
        
        if not features_list:
            print("No valid sessions found")
            return False
        
        # Create DataFrame
        self.features_df = pd.DataFrame(features_list, index=session_names)
        self.labels = np.array(labels_list)
        self.feature_names = list(features_list[0].keys())
        
        print(f"Loaded {len(features_list)} sessions with {len(self.feature_names)} features")
        return True
    
    def preprocess_data(self, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Preprocess the data for training.
        
        Args:
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if self.features_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Split features and labels
        X = self.features_df.values
        y = self.labels
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def get_feature_importance_data(self) -> pd.DataFrame:
        """
        Get feature importance data for analysis.
        
        Returns:
            DataFrame with feature statistics
        """
        if self.features_df is None:
            return pd.DataFrame()
        
        # Calculate feature statistics
        feature_stats = self.features_df.describe()
        
        # Add correlation with stress level
        correlations = []
        for feature in self.feature_names:
            corr = np.corrcoef(self.features_df[feature], self.labels)[0, 1]
            correlations.append(abs(corr))
        
        feature_stats.loc['correlation'] = correlations
        
        return feature_stats
    
    def get_session_summary(self) -> Dict:
        """
        Get summary statistics about the dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        if self.features_df is None:
            return {}
        
        # Label distribution
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        label_distribution = dict(zip(unique_labels, counts))
        
        # Feature statistics
        feature_stats = {
            'mean': self.features_df.mean().to_dict(),
            'std': self.features_df.std().to_dict(),
            'min': self.features_df.min().to_dict(),
            'max': self.features_df.max().to_dict()
        }
        
        return {
            'total_sessions': len(self.features_df),
            'feature_count': len(self.feature_names),
            'label_distribution': label_distribution,
            'feature_stats': feature_stats
        }
    
    def save_dataset(self, output_file: str) -> None:
        """
        Save the processed dataset to file.
        
        Args:
            output_file: Path to save the dataset
        """
        if self.features_df is None:
            raise ValueError("No data to save. Call load_data() first.")
        
        dataset_data = {
            'features': self.features_df.to_dict(),
            'labels': self.labels.tolist(),
            'feature_names': self.feature_names,
            'session_names': list(self.features_df.index)
        }
        
        with open(output_file, 'w') as f:
            json.dump(dataset_data, f, indent=2)
        
        print(f"Dataset saved to: {output_file}")
    
    def load_dataset(self, input_file: str) -> bool:
        """
        Load a previously saved dataset.
        
        Args:
            input_file: Path to the saved dataset
            
        Returns:
            True if dataset was loaded successfully
        """
        try:
            with open(input_file, 'r') as f:
                dataset_data = json.load(f)
            
            self.features_df = pd.DataFrame.from_dict(dataset_data['features'])
            self.labels = np.array(dataset_data['labels'])
            self.feature_names = dataset_data['feature_names']
            
            print(f"Dataset loaded from: {input_file}")
            print(f"Loaded {len(self.features_df)} sessions with {len(self.feature_names)} features")
            return True
        
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False
    
    def create_synthetic_data(self, n_samples: int = 100) -> None:
        """
        Create synthetic data for testing and development.
        
        Args:
            n_samples: Number of synthetic samples to create
        """
        # Define realistic feature ranges based on research
        feature_ranges = {
            'avg_interval': (0.1, 0.5),
            'median_interval': (0.08, 0.4),
            'cpm': (80, 300),
            'wpm': (16, 60),
            'error_rate': (0.02, 0.25),
            'backspace_rate': (0.01, 0.2),
            'delete_rate': (0.005, 0.1),
            'corrections_per_100': (2, 25),
            'pause_count': (0, 50),
            'pause_rate': (0.05, 0.4),
            'avg_pause_duration': (0.5, 3.0),
            'long_pause_rate': (0.01, 0.15),
            'interval_std': (0.05, 0.3),
            'interval_cv': (0.3, 1.2),
            'interval_range': (0.1, 1.0),
            'interval_iqr': (0.05, 0.25),
            'burst_count': (0, 30),
            'avg_burst_length': (2, 8),
            'burst_rate': (0.1, 0.5)
        }
        
        # Generate synthetic features
        features_list = []
        labels_list = []
        
        for i in range(n_samples):
            # Generate features within realistic ranges
            features = {}
            for feature, (min_val, max_val) in feature_ranges.items():
                features[feature] = np.random.uniform(min_val, max_val)
            
            # Generate stress level (1-3) with some correlation to features
            stress_score = (
                features['error_rate'] * 0.3 +
                features['pause_rate'] * 0.3 +
                (1 - features['cpm'] / 300) * 0.2 +
                features['interval_cv'] * 0.2
            )
            
            # Map to stress levels
            if stress_score < 0.3:
                stress_level = 1
            elif stress_score < 0.6:
                stress_level = 2
            else:
                stress_level = 3
            
            features_list.append(features)
            labels_list.append(stress_level)
        
        # Create DataFrame
        self.features_df = pd.DataFrame(features_list)
        self.labels = np.array(labels_list)
        self.feature_names = list(feature_ranges.keys())
        
        print(f"Created {n_samples} synthetic samples")


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="KeyStress Dataset Preparation")
    parser.add_argument("--load", action="store_true", help="Load existing dataset")
    parser.add_argument("--save", type=str, help="Save dataset to file")
    parser.add_argument("--synthetic", type=int, help="Create synthetic dataset with N samples")
    parser.add_argument("--summary", action="store_true", help="Show dataset summary")
    
    args = parser.parse_args()
    
    dataset = Dataset()
    
    if args.synthetic:
        dataset.create_synthetic_data(args.synthetic)
        print("Synthetic dataset created")
    
    elif args.load:
        if dataset.load_data():
            print("Dataset loaded successfully")
        else:
            print("Failed to load dataset")
    
    if args.save and dataset.features_df is not None:
        dataset.save_dataset(args.save)
    
    if args.summary and dataset.features_df is not None:
        summary = dataset.get_session_summary()
        print("Dataset Summary:")
        print(f"  Total sessions: {summary['total_sessions']}")
        print(f"  Features: {summary['feature_count']}")
        print(f"  Label distribution: {summary['label_distribution']}")


if __name__ == "__main__":
    main()
