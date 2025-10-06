"""
Model evaluation for KeyStress.

This module provides comprehensive evaluation metrics and cross-validation
for stress detection models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import json
import os


class ModelEvaluator:
    """
    Comprehensive model evaluator for stress detection.
    
    This class provides:
    - Standard classification metrics
    - ROC and PR curves
    - Cross-validation
    - Feature importance analysis
    - Confusion matrix visualization
    """
    
    def __init__(self):
        """Initialize the model evaluator."""
        pass
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray, 
                      feature_names: List[str] = None) -> Dict:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained model with predict and predict_proba methods
            X_test: Test features
            y_test: Test labels
            feature_names: Names of features
            
        Returns:
            Dictionary with comprehensive evaluation results
        """
        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_proba)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Feature importance (if available)
        feature_importance = self._get_feature_importance(model, feature_names)
        
        # ROC curves for each class
        roc_curves = self._calculate_roc_curves(y_test, y_proba)
        
        # Precision-Recall curves
        pr_curves = self._calculate_pr_curves(y_test, y_proba)
        
        return {
            'metrics': metrics,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'feature_importance': feature_importance,
            'roc_curves': roc_curves,
            'pr_curves': pr_curves,
            'predictions': y_pred.tolist(),
            'probabilities': y_proba.tolist()
        }
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_proba: np.ndarray) -> Dict:
        """Calculate comprehensive classification metrics."""
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(y_true, y_pred, average='macro')
        recall_macro = recall_score(y_true, y_pred, average='macro')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        # ROC AUC (one-vs-rest)
        try:
            roc_auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
        except:
            roc_auc = None
        
        return {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'roc_auc': roc_auc
        }
    
    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importance from model."""
        if feature_names is None:
            return {}
        
        if hasattr(model, 'feature_importances_'):
            # Random Forest
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models
            importance = np.abs(model.coef_[0])
        else:
            return {}
        
        return dict(zip(feature_names, importance))
    
    def _calculate_roc_curves(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict:
        """Calculate ROC curves for each class."""
        n_classes = y_proba.shape[1]
        roc_data = {}
        
        for i in range(n_classes):
            # One-vs-rest approach
            y_binary = (y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(y_binary, y_proba[:, i])
            auc = roc_auc_score(y_binary, y_proba[:, i])
            
            roc_data[f'class_{i}'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'auc': auc
            }
        
        return roc_data
    
    def _calculate_pr_curves(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict:
        """Calculate Precision-Recall curves for each class."""
        n_classes = y_proba.shape[1]
        pr_data = {}
        
        for i in range(n_classes):
            # One-vs-rest approach
            y_binary = (y_true == i).astype(int)
            precision, recall, _ = precision_recall_curve(y_binary, y_proba[:, i])
            
            pr_data[f'class_{i}'] = {
                'precision': precision.tolist(),
                'recall': recall.tolist()
            }
        
        return pr_data
    
    def cross_validate_model(self, model, X: np.ndarray, y: np.ndarray, 
                           cv_folds: int = 5) -> Dict:
        """
        Perform cross-validation on the model.
        
        Args:
            model: Model to cross-validate
            X: Features
            y: Labels
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with cross-validation results
        """
        # Define cross-validation strategy
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Cross-validation scores - use the underlying sklearn model
        if hasattr(model, 'model'):
            # Use the underlying sklearn model for cross-validation
            cv_scores = cross_val_score(model.model, X, y, cv=cv, scoring='accuracy')
        else:
            # Fallback: use the wrapper model
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        # Additional metrics
        if hasattr(model, 'model'):
            cv_precision = cross_val_score(model.model, X, y, cv=cv, scoring='precision_macro')
            cv_recall = cross_val_score(model.model, X, y, cv=cv, scoring='recall_macro')
            cv_f1 = cross_val_score(model.model, X, y, cv=cv, scoring='f1_macro')
        else:
            cv_precision = cross_val_score(model, X, y, cv=cv, scoring='precision_macro')
            cv_recall = cross_val_score(model, X, y, cv=cv, scoring='recall_macro')
            cv_f1 = cross_val_score(model, X, y, cv=cv, scoring='f1_macro')
        
        return {
            'cv_accuracy': {
                'scores': cv_scores.tolist(),
                'mean': cv_scores.mean(),
                'std': cv_scores.std()
            },
            'cv_precision': {
                'scores': cv_precision.tolist(),
                'mean': cv_precision.mean(),
                'std': cv_precision.std()
            },
            'cv_recall': {
                'scores': cv_recall.tolist(),
                'mean': cv_recall.mean(),
                'std': cv_recall.std()
            },
            'cv_f1': {
                'scores': cv_f1.tolist(),
                'mean': cv_f1.mean(),
                'std': cv_f1.std()
            }
        }
    
    def plot_evaluation_results(self, results: Dict, save_path: str = None) -> None:
        """
        Create comprehensive evaluation plots.
        
        Args:
            results: Evaluation results dictionary
            save_path: Path to save plots (optional)
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Confusion Matrix
        cm = np.array(results['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # 2. Metrics Bar Plot
        metrics = results['metrics']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']
        metric_values = [
            metrics['accuracy'],
            metrics['precision_macro'],
            metrics['recall_macro'],
            metrics['f1_macro']
        ]
        
        axes[0, 1].bar(metric_names, metric_values, color=['blue', 'green', 'orange', 'red'])
        axes[0, 1].set_title('Model Performance Metrics')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_ylim(0, 1)
        
        # 3. Feature Importance
        if results['feature_importance']:
            importance_data = results['feature_importance']
            features = list(importance_data.keys())
            importance_values = list(importance_data.values())
            
            # Sort by importance
            sorted_indices = np.argsort(importance_values)[::-1]
            top_features = [features[i] for i in sorted_indices[:10]]
            top_values = [importance_values[i] for i in sorted_indices[:10]]
            
            axes[0, 2].barh(range(len(top_features)), top_values)
            axes[0, 2].set_yticks(range(len(top_features)))
            axes[0, 2].set_yticklabels(top_features)
            axes[0, 2].set_title('Top 10 Feature Importance')
            axes[0, 2].set_xlabel('Importance')
        
        # 4. ROC Curves
        roc_curves = results['roc_curves']
        for class_name, roc_data in roc_curves.items():
            fpr = roc_data['fpr']
            tpr = roc_data['tpr']
            auc = roc_data['auc']
            axes[1, 0].plot(fpr, tpr, label=f'{class_name} (AUC = {auc:.3f})')
        
        axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[1, 0].set_xlabel('False Positive Rate')
        axes[1, 0].set_ylabel('True Positive Rate')
        axes[1, 0].set_title('ROC Curves')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Precision-Recall Curves
        pr_curves = results['pr_curves']
        for class_name, pr_data in pr_curves.items():
            precision = pr_data['precision']
            recall = pr_data['recall']
            axes[1, 1].plot(recall, precision, label=class_name)
        
        axes[1, 1].set_xlabel('Recall')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].set_title('Precision-Recall Curves')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Per-class Performance
        metrics = results['metrics']
        classes = ['Low', 'Medium', 'High']
        precision_per_class = metrics['precision_per_class']
        recall_per_class = metrics['recall_per_class']
        f1_per_class = metrics['f1_per_class']
        
        x = np.arange(len(classes))
        width = 0.25
        
        axes[1, 2].bar(x - width, precision_per_class, width, label='Precision', alpha=0.8)
        axes[1, 2].bar(x, recall_per_class, width, label='Recall', alpha=0.8)
        axes[1, 2].bar(x + width, f1_per_class, width, label='F1', alpha=0.8)
        
        axes[1, 2].set_xlabel('Stress Level')
        axes[1, 2].set_ylabel('Score')
        axes[1, 2].set_title('Per-Class Performance')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(classes)
        axes[1, 2].legend()
        axes[1, 2].set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Evaluation plots saved to: {save_path}")
        
        plt.show()
    
    def generate_evaluation_report(self, results: Dict, model_name: str = "Model") -> str:
        """
        Generate a text-based evaluation report.
        
        Args:
            results: Evaluation results
            model_name: Name of the model
            
        Returns:
            Formatted evaluation report
        """
        metrics = results['metrics']
        
        report = f"""
{'='*60}
EVALUATION REPORT: {model_name}
{'='*60}

OVERALL PERFORMANCE:
- Accuracy: {metrics['accuracy']:.3f}
- Macro Precision: {metrics['precision_macro']:.3f}
- Macro Recall: {metrics['recall_macro']:.3f}
- Macro F1-Score: {metrics['f1_macro']:.3f}
        - ROC AUC: {f"{metrics['roc_auc']:.3f}" if metrics['roc_auc'] is not None else 'N/A'}

PER-CLASS PERFORMANCE:
"""
        
        classes = ['Low Stress', 'Medium Stress', 'High Stress']
        for i, class_name in enumerate(classes):
            report += f"- {class_name}:\n"
            report += f"  - Precision: {metrics['precision_per_class'][i]:.3f}\n"
            report += f"  - Recall: {metrics['recall_per_class'][i]:.3f}\n"
            report += f"  - F1-Score: {metrics['f1_per_class'][i]:.3f}\n"
        
        # Feature importance
        if results['feature_importance']:
            report += "\nTOP FEATURES BY IMPORTANCE:\n"
            importance_data = results['feature_importance']
            sorted_features = sorted(importance_data.items(), key=lambda x: x[1], reverse=True)
            
            for i, (feature, importance) in enumerate(sorted_features[:5]):
                report += f"- {feature}: {importance:.3f}\n"
        
        report += f"\n{'='*60}\n"
        
        return report
    
    def save_evaluation_results(self, results: Dict, output_file: str) -> None:
        """
        Save evaluation results to file.
        
        Args:
            results: Evaluation results
            output_file: Path to save results
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, dict):
                serializable_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        serializable_results[key][k] = v.tolist()
                    else:
                        serializable_results[key][k] = v
            else:
                serializable_results[key] = value
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Evaluation results saved to: {output_file}")


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="KeyStress Model Evaluation")
    parser.add_argument("--model-file", type=str, required=True, help="Path to trained model")
    parser.add_argument("--test-data", type=str, required=True, help="Path to test data")
    parser.add_argument("--output", type=str, help="Output file for results")
    parser.add_argument("--plot", action="store_true", help="Generate evaluation plots")
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator()
    
    # Load model and test data
    # This would need to be implemented based on your model loading mechanism
    print("Model evaluation functionality requires model and data loading implementation")


if __name__ == "__main__":
    main()
