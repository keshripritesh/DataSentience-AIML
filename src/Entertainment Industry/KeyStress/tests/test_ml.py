"""
Tests for machine learning components.

This module tests the ML functionality including dataset preparation,
model training, and evaluation.
"""

import pytest
import numpy as np
import pandas as pd
import json
import tempfile
import os
from unittest.mock import Mock, patch

from ml.dataset import Dataset
from ml.model import StressModel
from ml.train import ModelTrainer
from ml.evaluate import ModelEvaluator


class TestDataset:
    """Test cases for Dataset class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.dataset = Dataset()
    
    def test_create_synthetic_data(self):
        """Test synthetic data creation."""
        self.dataset.create_synthetic_data(n_samples=10)
        
        assert self.dataset.features_df is not None
        assert len(self.dataset.features_df) == 10
        assert len(self.dataset.labels) == 10
        assert len(self.dataset.feature_names) > 0
        
        # Check that features are within reasonable ranges
        for feature in self.dataset.feature_names:
            values = self.dataset.features_df[feature]
            assert values.min() >= 0  # Most features should be non-negative
    
    def test_preprocess_data(self):
        """Test data preprocessing."""
        # Create synthetic data first
        self.dataset.create_synthetic_data(n_samples=20)
        
        # Preprocess data
        X_train, X_test, y_train, y_test = self.dataset.preprocess_data(test_size=0.3)
        
        # Check shapes
        assert len(X_train) + len(X_test) == len(self.dataset.features_df)
        assert len(y_train) + len(y_test) == len(self.dataset.labels)
        
        # Check that data is scaled
        assert np.allclose(X_train.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(X_train.std(axis=0), 1, atol=1e-10)
    
    def test_get_session_summary(self):
        """Test session summary generation."""
        # Create synthetic data
        self.dataset.create_synthetic_data(n_samples=15)
        
        summary = self.dataset.get_session_summary()
        
        expected_keys = ['total_sessions', 'feature_count', 'label_distribution', 'feature_stats']
        
        for key in expected_keys:
            assert key in summary
        
        assert summary['total_sessions'] == 15
        assert summary['feature_count'] == len(self.dataset.feature_names)
    
    def test_save_and_load_dataset(self):
        """Test dataset saving and loading."""
        # Create synthetic data
        self.dataset.create_synthetic_data(n_samples=5)
        
        # Save dataset
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            self.dataset.save_dataset(temp_file)
            
            # Create new dataset and load
            new_dataset = Dataset()
            success = new_dataset.load_dataset(temp_file)
            
            assert success
            assert len(new_dataset.features_df) == 5
            assert len(new_dataset.labels) == 5
            assert new_dataset.feature_names == self.dataset.feature_names
            
        finally:
            os.unlink(temp_file)
    
    def test_get_feature_importance_data(self):
        """Test feature importance data generation."""
        # Create synthetic data
        self.dataset.create_synthetic_data(n_samples=10)
        
        feature_stats = self.dataset.get_feature_importance_data()
        
        assert not feature_stats.empty
        assert 'correlation' in feature_stats.index
        
        # Check that correlations are between -1 and 1
        correlations = feature_stats.loc['correlation']
        assert np.all(correlations >= 0)  # We use absolute correlations
        assert np.all(correlations <= 1)


class TestStressModel:
    """Test cases for StressModel class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create sample data
        np.random.seed(42)
        self.X = np.random.randn(100, 19)  # 100 samples, 19 features
        self.y = np.random.randint(0, 3, 100)  # 3 classes
        self.feature_names = [f'feature_{i}' for i in range(19)]
    
    def test_model_initialization(self):
        """Test model initialization."""
        # Test different model types
        model_types = ['logistic', 'random_forest', 'svm']
        
        for model_type in model_types:
            model = StressModel(model_type=model_type)
            assert model.model_type == model_type
            assert model.model is not None
            assert not model.is_trained
    
    def test_model_training(self):
        """Test model training."""
        model = StressModel(model_type='random_forest')
        
        # Train model
        results = model.train(self.X, self.y, self.feature_names, ['Low', 'Medium', 'High'])
        
        assert model.is_trained
        assert 'training_score' in results
        assert 'feature_importance' in results
        assert model.feature_names == self.feature_names
    
    def test_model_prediction(self):
        """Test model prediction."""
        model = StressModel(model_type='random_forest')
        model.train(self.X, self.y, self.feature_names)
        
        # Test predictions
        X_test = np.random.randn(10, 19)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        assert len(predictions) == 10
        assert predictions.shape == (10,)
        assert probabilities.shape == (10, 3)  # 3 classes
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    
    def test_model_evaluation(self):
        """Test model evaluation."""
        model = StressModel(model_type='random_forest')
        model.train(self.X, self.y, self.feature_names)
        
        # Split data for evaluation
        X_test = self.X[:20]
        y_test = self.y[:20]
        
        results = model.evaluate(X_test, y_test)
        
        expected_keys = ['accuracy', 'classification_report', 'confusion_matrix', 
                        'predictions', 'probabilities']
        
        for key in expected_keys:
            assert key in results
        
        assert 0 <= results['accuracy'] <= 1
    
    def test_model_save_load(self):
        """Test model saving and loading."""
        model = StressModel(model_type='random_forest')
        model.train(self.X, self.y, self.feature_names)
        
        # Save model
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_file = f.name
        
        try:
            model.save_model(temp_file)
            
            # Load model
            new_model = StressModel(model_type='random_forest')
            new_model.load_model(temp_file)
            
            assert new_model.is_trained
            assert new_model.feature_names == model.feature_names
            
            # Test that predictions are the same
            X_test = np.random.randn(5, 19)
            pred1 = model.predict(X_test)
            pred2 = new_model.predict(X_test)
            
            np.testing.assert_array_equal(pred1, pred2)
            
        finally:
            # Clean up files
            for ext in ['.joblib', '_metadata.json']:
                try:
                    os.unlink(temp_file + ext)
                except FileNotFoundError:
                    pass
    
    def test_invalid_model_type(self):
        """Test handling of invalid model type."""
        with pytest.raises(ValueError):
            StressModel(model_type='invalid_model')
    
    def test_prediction_before_training(self):
        """Test that prediction fails before training."""
        model = StressModel(model_type='random_forest')
        
        with pytest.raises(ValueError):
            model.predict(np.random.randn(5, 19))
        
        with pytest.raises(ValueError):
            model.predict_proba(np.random.randn(5, 19))


class TestModelTrainer:
    """Test cases for ModelTrainer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.trainer = ModelTrainer()
    
    def test_prepare_data(self):
        """Test data preparation."""
        # This will create synthetic data since no real data exists
        success = self.trainer.prepare_data(min_sessions=1)
        
        assert success
        assert hasattr(self.trainer, 'X_train')
        assert hasattr(self.trainer, 'X_test')
        assert hasattr(self.trainer, 'y_train')
        assert hasattr(self.trainer, 'y_test')
    
    def test_train_model(self):
        """Test model training."""
        # Prepare data first
        self.trainer.prepare_data(min_sessions=1)
        
        # Train model
        results = self.trainer.train_model(model_type='random_forest')
        
        assert 'model_type' in results
        assert 'training_results' in results
        assert 'evaluation_results' in results
        assert results['model_type'] == 'random_forest'
        
        # Check that model was saved
        assert 'random_forest' in self.trainer.training_history
    
    def test_train_multiple_models(self):
        """Test training multiple models."""
        # Prepare data first
        self.trainer.prepare_data(min_sessions=1)
        
        # Train multiple models
        results = self.trainer.train_multiple_models(['logistic', 'random_forest'])
        
        assert len(results) == 2
        assert 'logistic' in results
        assert 'random_forest' in results
        
        # Check that all models have evaluation results
        for model_type, result in results.items():
            assert 'evaluation_results' in result
            assert 'accuracy' in result['evaluation_results']
    
    def test_hyperparameter_tuning(self):
        """Test hyperparameter tuning."""
        # Prepare data first
        self.trainer.prepare_data(min_sessions=1)
        
        # Train model with hyperparameter tuning
        results = self.trainer.train_model(
            model_type='random_forest',
            hyperparameter_tuning=True
        )
        
        assert 'model_type' in results
        assert 'training_results' in results
        assert 'evaluation_results' in results
    
    def test_save_training_results(self):
        """Test saving training results."""
        # Prepare data and train model
        self.trainer.prepare_data(min_sessions=1)
        self.trainer.train_model(model_type='random_forest')
        
        # Save results
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            self.trainer.save_training_results(temp_file)
            
            # Check that file was created
            assert os.path.exists(temp_file)
            
            # Load and verify
            with open(temp_file, 'r') as f:
                data = json.load(f)
            
            assert 'random_forest' in data
            
        finally:
            os.unlink(temp_file)


class TestModelEvaluator:
    """Test cases for ModelEvaluator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = ModelEvaluator()
        
        # Create sample data
        np.random.seed(42)
        self.X = np.random.randn(100, 19)
        self.y = np.random.randint(0, 3, 100)
        self.feature_names = [f'feature_{i}' for i in range(19)]
    
    def test_evaluate_model(self):
        """Test model evaluation."""
        # Create and train a model
        model = StressModel(model_type='random_forest')
        model.train(self.X, self.y, self.feature_names)
        
        # Evaluate model
        X_test = self.X[:20]
        y_test = self.y[:20]
        
        results = self.evaluator.evaluate_model(model, X_test, y_test, self.feature_names)
        
        expected_keys = ['metrics', 'confusion_matrix', 'classification_report',
                        'feature_importance', 'roc_curves', 'pr_curves',
                        'predictions', 'probabilities']
        
        for key in expected_keys:
            assert key in results
        
        # Check metrics
        metrics = results['metrics']
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision_macro'] <= 1
        assert 0 <= metrics['recall_macro'] <= 1
        assert 0 <= metrics['f1_macro'] <= 1
    
    def test_cross_validate_model(self):
        """Test cross-validation."""
        # Create a model
        model = StressModel(model_type='random_forest')
        
        # Perform cross-validation
        cv_results = self.evaluator.cross_validate_model(model, self.X, self.y, cv_folds=3)
        
        expected_keys = ['cv_accuracy', 'cv_precision', 'cv_recall', 'cv_f1']
        
        for key in expected_keys:
            assert key in cv_results
            assert 'scores' in cv_results[key]
            assert 'mean' in cv_results[key]
            assert 'std' in cv_results[key]
        
        # Check that we have the right number of scores
        assert len(cv_results['cv_accuracy']['scores']) == 3
    
    def test_generate_evaluation_report(self):
        """Test evaluation report generation."""
        # Create sample results
        sample_results = {
            'metrics': {
                'accuracy': 0.85,
                'precision_macro': 0.83,
                'recall_macro': 0.85,
                'f1_macro': 0.84,
                'precision_per_class': [0.9, 0.8, 0.8],
                'recall_per_class': [0.9, 0.8, 0.85],
                'f1_per_class': [0.9, 0.8, 0.82],
                'roc_auc': 0.92
            },
            'feature_importance': {
                'feature_1': 0.3,
                'feature_2': 0.2,
                'feature_3': 0.1
            }
        }
        
        report = self.evaluator.generate_evaluation_report(sample_results, "Test Model")
        
        assert isinstance(report, str)
        assert "Test Model" in report
        assert "0.850" in report  # Accuracy
        assert "feature_1" in report  # Feature importance
    
    def test_save_evaluation_results(self):
        """Test saving evaluation results."""
        # Create sample results
        sample_results = {
            'metrics': {
                'accuracy': 0.85,
                'precision_macro': 0.83
            },
            'confusion_matrix': [[10, 2, 1], [1, 12, 1], [1, 1, 11]],
            'feature_importance': {'feature_1': 0.3, 'feature_2': 0.2}
        }
        
        # Save results
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            self.evaluator.save_evaluation_results(sample_results, temp_file)
            
            # Check that file was created and contains data
            assert os.path.exists(temp_file)
            
            with open(temp_file, 'r') as f:
                data = json.load(f)
            
            assert 'metrics' in data
            assert 'confusion_matrix' in data
            assert 'feature_importance' in data
            
        finally:
            os.unlink(temp_file)


class TestMLIntegration:
    """Integration tests for ML components."""
    
    def test_end_to_end_ml_pipeline(self):
        """Test complete ML pipeline from data to evaluation."""
        # Create dataset
        dataset = Dataset()
        dataset.create_synthetic_data(n_samples=50)
        
        # Preprocess data
        X_train, X_test, y_train, y_test = dataset.preprocess_data(test_size=0.2)
        
        # Train model
        model = StressModel(model_type='random_forest')
        model.train(X_train, y_train, dataset.feature_names)
        
        # Evaluate model
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_model(model, X_test, y_test, dataset.feature_names)
        
        # Verify results
        assert 'metrics' in results
        assert 'accuracy' in results['metrics']
        assert 0 <= results['metrics']['accuracy'] <= 1
        
        # Test predictions
        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test)
        
        # Test probabilities
        probabilities = model.predict_proba(X_test)
        # Check that probabilities sum to 1 for each sample
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        # Check that we have the right number of classes
        n_classes = len(np.unique(y_train))
        assert probabilities.shape == (len(X_test), n_classes)
    
    def test_model_comparison(self):
        """Test comparing different model types."""
        # Create dataset
        dataset = Dataset()
        dataset.create_synthetic_data(n_samples=30)
        
        # Preprocess data
        X_train, X_test, y_train, y_test = dataset.preprocess_data(test_size=0.3)
        
        # Test different model types
        model_types = ['logistic', 'random_forest']
        results = {}
        
        for model_type in model_types:
            model = StressModel(model_type=model_type)
            model.train(X_train, y_train, dataset.feature_names)
            
            evaluator = ModelEvaluator()
            eval_results = evaluator.evaluate_model(model, X_test, y_test, dataset.feature_names)
            
            results[model_type] = eval_results['metrics']['accuracy']
        
        # Verify that we got results for all models
        assert len(results) == 2
        assert all(0 <= acc <= 1 for acc in results.values())


if __name__ == "__main__":
    pytest.main([__file__])
