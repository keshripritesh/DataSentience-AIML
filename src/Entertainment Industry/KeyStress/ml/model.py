"""
Machine learning models for stress detection.

This module provides various ML models for stress detection from keystroke data,
including Logistic Regression, Random Forest, and LSTM models.
"""

import numpy as np
import pandas as pd
import pickle
import json
from typing import Dict, List, Tuple, Optional, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class StressModel:
    """
    Base class for stress detection models.
    
    This class provides a common interface for different ML models
    used for stress detection from keystroke features.
    """
    
    def __init__(self, model_type: str = "random_forest", **kwargs):
        """
        Initialize the stress model.
        
        Args:
            model_type: Type of model to use ('logistic', 'random_forest', 'svm', 'lstm')
            **kwargs: Additional arguments for the specific model
        """
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        self.class_names = None
        self.is_trained = False
        
        # Initialize the specific model
        self._initialize_model(**kwargs)
    
    def _initialize_model(self, **kwargs):
        """Initialize the specific model based on model_type."""
        if self.model_type == "logistic":
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                **kwargs
            )
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                **kwargs
            )
        elif self.model_type == "svm":
            self.model = SVC(
                probability=True,
                random_state=42,
                **kwargs
            )
        elif self.model_type == "lstm" and TORCH_AVAILABLE:
            self.model = LSTMStressModel(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              feature_names: List[str] = None, class_names: List[str] = None) -> Dict:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            feature_names: Names of features
            class_names: Names of classes
            
        Returns:
            Dictionary with training results
        """
        self.feature_names = feature_names
        self.class_names = class_names
        
        # Train the model
        if self.model_type == "lstm":
            results = self.model.train(X_train, y_train)
        else:
            self.model.fit(X_train, y_train)
            results = {
                'training_score': self.model.score(X_train, y_train),
                'feature_importance': self._get_feature_importance()
            }
        
        self.is_trained = True
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if self.model_type == "lstm":
            return self.model.predict(X)
        else:
            return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Features to predict on
            
        Returns:
            Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if self.model_type == "lstm":
            return self.model.predict_proba(X)
        elif hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # For models without predict_proba, return one-hot encoded predictions
            predictions = self.predict(X)
            n_classes = len(np.unique(predictions))
            proba = np.zeros((len(X), n_classes))
            for i, pred in enumerate(predictions):
                proba[i, pred] = 1.0
            return proba
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate the model.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y_test)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': y_pred.tolist(),
            'probabilities': y_proba.tolist()
        }
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance if available."""
        if not self.is_trained or self.feature_names is None:
            return {}
        
        if hasattr(self.model, 'feature_importances_'):
            # Random Forest
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # Logistic Regression
            importance = np.abs(self.model.coef_[0])
        else:
            return {}
        
        return dict(zip(self.feature_names, importance))
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to file.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Save model metadata
        metadata = {
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'is_trained': self.is_trained
        }
        
        # Save the model
        if self.model_type == "lstm":
            self.model.save_model(filepath, metadata)
        else:
            # Save scikit-learn model
            joblib.dump(self.model, f"{filepath}.joblib")
            
            # Save metadata
            with open(f"{filepath}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from file.
        
        Args:
            filepath: Path to the saved model
        """
        if self.model_type == "lstm":
            self.model.load_model(filepath)
        else:
            # Load scikit-learn model
            self.model = joblib.load(f"{filepath}.joblib")
            
            # Load metadata
            with open(f"{filepath}_metadata.json", 'r') as f:
                metadata = json.load(f)
            
            self.feature_names = metadata['feature_names']
            self.class_names = metadata['class_names']
            self.is_trained = metadata['is_trained']


class LSTMStressModel:
    """
    LSTM model for stress detection from sequential keystroke data.
    
    This model is designed to capture temporal patterns in keystroke sequences.
    """
    
    def __init__(self, input_size: int = 19, hidden_size: int = 64, 
                 num_layers: int = 2, num_classes: int = 3, dropout: float = 0.2):
        """
        Initialize the LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layers
            num_layers: Number of LSTM layers
            num_classes: Number of stress classes
            dropout: Dropout rate
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for LSTM models")
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        
        # Initialize the neural network
        self.network = LSTMNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout
        )
        
        self.optimizer = torch.optim.Adam(self.network.parameters())
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network.to(self.device)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              epochs: int = 100, batch_size: int = 32) -> Dict:
        """
        Train the LSTM model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary with training results
        """
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        y_tensor = torch.LongTensor(y_train).to(self.device)
        
        # Training loop
        losses = []
        accuracies = []
        
        for epoch in range(epochs):
            self.network.train()
            
            # Forward pass
            outputs = self.network(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y_tensor).sum().item() / y_tensor.size(0)
            
            losses.append(loss.item())
            accuracies.append(accuracy)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Accuracy = {accuracy:.4f}")
        
        return {
            'final_loss': losses[-1],
            'final_accuracy': accuracies[-1],
            'losses': losses,
            'accuracies': accuracies
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        self.network.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.network(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
            return predicted.cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        self.network.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.network(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            return probabilities.cpu().numpy()
    
    def save_model(self, filepath: str, metadata: Dict) -> None:
        """Save the LSTM model."""
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metadata': metadata
        }, f"{filepath}.pth")
    
    def load_model(self, filepath: str) -> None:
        """Load the LSTM model."""
        checkpoint = torch.load(f"{filepath}.pth", map_location=self.device)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class LSTMNetwork(nn.Module):
    """LSTM neural network for stress detection."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 num_classes: int, dropout: float = 0.2):
        super(LSTMNetwork, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Get the last output
        out = out[:, -1, :]
        
        # Fully connected layers
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="KeyStress Model")
    parser.add_argument("--model-type", type=str, default="random_forest", 
                       choices=["logistic", "random_forest", "svm", "lstm"],
                       help="Type of model to use")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--predict", type=str, help="Make predictions on features file")
    parser.add_argument("--save", type=str, help="Save model to file")
    parser.add_argument("--load", type=str, help="Load model from file")
    
    args = parser.parse_args()
    
    model = StressModel(model_type=args.model_type)
    
    if args.load:
        model.load_model(args.load)
        print(f"Model loaded from {args.load}")
    
    if args.save:
        model.save_model(args.save)
        print(f"Model saved to {args.save}")
    
    if args.predict:
        # Load features and make predictions
        import json
        with open(args.predict, 'r') as f:
            features = json.load(f)
        
        # Convert to numpy array
        X = np.array([list(features.values())])
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        print(f"Prediction: {predictions[0]}")
        print(f"Probabilities: {probabilities[0]}")


if __name__ == "__main__":
    main()
