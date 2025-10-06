"""
Model training for KeyStress.

This module handles the training loop for stress detection models,
including data preparation, model training, and validation.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from ml.dataset import Dataset
from ml.model import StressModel
from ml.evaluate import ModelEvaluator


class ModelTrainer:
    """
    Trainer for stress detection models.
    
    This class handles:
    - Data preparation and preprocessing
    - Model training with cross-validation
    - Hyperparameter tuning
    - Model evaluation and saving
    """
    
    def __init__(self, data_dir: str = "data", models_dir: str = "models"):
        """
        Initialize the model trainer.
        
        Args:
            data_dir: Directory containing training data
            models_dir: Directory to save trained models
        """
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.dataset = Dataset(data_dir)
        self.evaluator = ModelEvaluator()
        
        # Ensure models directory exists
        os.makedirs(models_dir, exist_ok=True)
        
        # Training history
        self.training_history = {}
    
    def prepare_data(self, min_sessions: int = 5, test_size: float = 0.2) -> bool:
        """
        Prepare training data.
        
        Args:
            min_sessions: Minimum number of sessions required
            test_size: Fraction of data to use for testing
            
        Returns:
            True if data preparation was successful
        """
        print("Loading and preparing data...")
        
        # Load data
        if not self.dataset.load_data(min_sessions=min_sessions):
            print("Failed to load sufficient data. Creating synthetic data for testing...")
            self.dataset.create_synthetic_data(n_samples=50)
        
        # Preprocess data
        try:
            self.X_train, self.X_test, self.y_train, self.y_test = self.dataset.preprocess_data(
                test_size=test_size
            )
            
            print(f"Data prepared successfully:")
            print(f"  Training samples: {len(self.X_train)}")
            print(f"  Test samples: {len(self.X_test)}")
            print(f"  Features: {len(self.dataset.feature_names)}")
            
            return True
        
        except Exception as e:
            print(f"Error preparing data: {e}")
            return False
    
    def train_model(self, model_type: str = "random_forest", 
                   hyperparameter_tuning: bool = False) -> Dict:
        """
        Train a stress detection model.
        
        Args:
            model_type: Type of model to train
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary with training results
        """
        if not hasattr(self, 'X_train'):
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        print(f"Training {model_type} model...")
        
        # Initialize model
        model = StressModel(model_type=model_type)
        
        # Hyperparameter tuning if requested
        if hyperparameter_tuning:
            model = self._tune_hyperparameters(model, model_type)
        
        # Train the model
        training_results = model.train(
            self.X_train, 
            self.y_train,
            feature_names=self.dataset.feature_names,
            class_names=['Low', 'Medium', 'High']
        )
        
        # Evaluate the model
        evaluation_results = model.evaluate(self.X_test, self.y_test)
        
        # Combine results
        results = {
            'model_type': model_type,
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'feature_names': self.dataset.feature_names,
            'dataset_summary': self.dataset.get_session_summary()
        }
        
        # Save the model
        model_path = os.path.join(self.models_dir, f"stress_model_{model_type}")
        model.save_model(model_path)
        
        # Store in training history
        self.training_history[model_type] = results
        
        print(f"Model training completed:")
        print(f"  Accuracy: {evaluation_results['accuracy']:.3f}")
        print(f"  Model saved to: {model_path}")
        
        return results
    
    def _tune_hyperparameters(self, model: StressModel, model_type: str) -> StressModel:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            model: Initial model
            model_type: Type of model
            
        Returns:
            Model with tuned hyperparameters
        """
        print("Performing hyperparameter tuning...")
        
        # Define parameter grids for different model types
        if model_type == "random_forest":
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif model_type == "logistic":
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        elif model_type == "svm":
            param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto', 0.1, 0.01]
            }
        else:
            print(f"Hyperparameter tuning not implemented for {model_type}")
            return model
        
        # Perform grid search
        grid_search = GridSearchCV(
            model.model,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
        
        # Update model with best parameters
        model.model = grid_search.best_estimator_
        
        return model
    
    def train_multiple_models(self, model_types: List[str] = None) -> Dict:
        """
        Train multiple models and compare their performance.
        
        Args:
            model_types: List of model types to train
            
        Returns:
            Dictionary with results for all models
        """
        if model_types is None:
            model_types = ["logistic", "random_forest", "svm"]
        
        results = {}
        
        for model_type in model_types:
            try:
                print(f"\n{'='*50}")
                print(f"Training {model_type.upper()} model")
                print(f"{'='*50}")
                
                result = self.train_model(model_type=model_type)
                results[model_type] = result
                
            except Exception as e:
                print(f"Error training {model_type} model: {e}")
                continue
        
        # Compare models
        self._compare_models(results)
        
        return results
    
    def _compare_models(self, results: Dict) -> None:
        """
        Compare the performance of different models.
        
        Args:
            results: Dictionary with results for all models
        """
        print(f"\n{'='*50}")
        print("MODEL COMPARISON")
        print(f"{'='*50}")
        
        comparison_data = []
        
        for model_type, result in results.items():
            accuracy = result['evaluation_results']['accuracy']
            comparison_data.append({
                'Model': model_type.upper(),
                'Accuracy': accuracy,
                'Training Score': result['training_results'].get('training_score', 0)
            })
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        print(comparison_df.to_string(index=False))
        
        # Plot comparison
        self._plot_model_comparison(comparison_df)
    
    def _plot_model_comparison(self, comparison_df: pd.DataFrame) -> None:
        """
        Create a plot comparing model performance.
        
        Args:
            comparison_df: DataFrame with model comparison data
        """
        plt.figure(figsize=(10, 6))
        
        # Bar plot of accuracies
        plt.subplot(1, 2, 1)
        plt.bar(comparison_df['Model'], comparison_df['Accuracy'])
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        # Training vs Test accuracy
        plt.subplot(1, 2, 2)
        plt.scatter(comparison_df['Training Score'], comparison_df['Accuracy'], 
                   s=100, alpha=0.7)
        for i, model in enumerate(comparison_df['Model']):
            plt.annotate(model, (comparison_df['Training Score'].iloc[i], 
                               comparison_df['Accuracy'].iloc[i]))
        plt.plot([0, 1], [0, 1], 'r--', alpha=0.5)
        plt.xlabel('Training Score')
        plt.ylabel('Test Accuracy')
        plt.title('Training vs Test Accuracy')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.models_dir, "model_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to: {plot_path}")
        
        # Don't show plot in test environment
        plt.close()
    
    def save_training_results(self, output_file: str = None) -> None:
        """
        Save training results to file.
        
        Args:
            output_file: Path to save results (optional)
        """
        if output_file is None:
            output_file = os.path.join(self.models_dir, "training_results.json")
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        
        for model_type, results in self.training_history.items():
            # Convert dataset summary to be JSON serializable
            dataset_summary = results['dataset_summary'].copy()
            if 'label_distribution' in dataset_summary:
                # Convert numpy int64 keys to regular int
                label_dist = {}
                for k, v in dataset_summary['label_distribution'].items():
                    label_dist[int(k)] = int(v)
                dataset_summary['label_distribution'] = label_dist
            
            serializable_results[model_type] = {
                'model_type': results['model_type'],
                'training_results': results['training_results'],
                'evaluation_results': results['evaluation_results'],
                'feature_names': results['feature_names'],
                'dataset_summary': dataset_summary
            }
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Training results saved to: {output_file}")
    
    def load_training_results(self, input_file: str) -> Dict:
        """
        Load training results from file.
        
        Args:
            input_file: Path to load results from
            
        Returns:
            Dictionary with training results
        """
        with open(input_file, 'r') as f:
            results = json.load(f)
        
        self.training_history = results
        return results


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="KeyStress Model Training")
    parser.add_argument("--model-type", type=str, default="random_forest",
                       choices=["logistic", "random_forest", "svm", "lstm"],
                       help="Type of model to train")
    parser.add_argument("--tune", action="store_true", help="Perform hyperparameter tuning")
    parser.add_argument("--compare", action="store_true", help="Train and compare multiple models")
    parser.add_argument("--min-sessions", type=int, default=5, help="Minimum sessions required")
    parser.add_argument("--output", type=str, help="Output file for results")
    
    args = parser.parse_args()
    
    trainer = ModelTrainer()
    
    # Prepare data
    if not trainer.prepare_data(min_sessions=args.min_sessions):
        print("Failed to prepare data")
        return
    
    # Train model(s)
    if args.compare:
        results = trainer.train_multiple_models()
    else:
        results = trainer.train_model(
            model_type=args.model_type,
            hyperparameter_tuning=args.tune
        )
    
    # Save results
    if args.output:
        trainer.save_training_results(args.output)
    else:
        trainer.save_training_results()


if __name__ == "__main__":
    main()
