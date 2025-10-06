"""
Evaluation Metrics for FluidNetSim.

Provides various metrics for evaluating fluid dynamics predictions.
"""

import numpy as np
import torch
from typing import Dict, Any, Union, Optional
import logging

logger = logging.getLogger(__name__)

class FluidMetrics:
    """
    Evaluation metrics for fluid dynamics predictions.
    
    Features:
    - MSE, MAE, RMSE
    - Physics-based metrics
    - Flow field analysis
    - Statistical measures
    """
    
    def __init__(self):
        """Initialize metrics calculator."""
        pass
    
    @staticmethod
    def mse(pred: Union[np.ndarray, torch.Tensor], 
            target: Union[np.ndarray, torch.Tensor]) -> float:
        """Mean Squared Error."""
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        
        return np.mean((pred - target) ** 2)
    
    @staticmethod
    def mae(pred: Union[np.ndarray, torch.Tensor], 
            target: Union[np.ndarray, torch.Tensor]) -> float:
        """Mean Absolute Error."""
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        
        return np.mean(np.abs(pred - target))
    
    @staticmethod
    def rmse(pred: Union[np.ndarray, torch.Tensor], 
             target: Union[np.ndarray, torch.Tensor]) -> float:
        """Root Mean Squared Error."""
        return np.sqrt(FluidMetrics.mse(pred, target))
    
    @staticmethod
    def relative_error(pred: Union[np.ndarray, torch.Tensor], 
                      target: Union[np.ndarray, torch.Tensor]) -> float:
        """Relative Error."""
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        
        return np.mean(np.abs(pred - target) / (np.abs(target) + 1e-8))
    
    @staticmethod
    def correlation_coefficient(pred: Union[np.ndarray, torch.Tensor], 
                              target: Union[np.ndarray, torch.Tensor]) -> float:
        """Pearson Correlation Coefficient."""
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        
        return np.corrcoef(pred_flat, target_flat)[0, 1]
    
    @staticmethod
    def divergence_error(velocity_x: Union[np.ndarray, torch.Tensor],
                        velocity_y: Union[np.ndarray, torch.Tensor]) -> float:
        """Compute divergence error for incompressible flow."""
        if isinstance(velocity_x, torch.Tensor):
            velocity_x = velocity_x.detach().cpu().numpy()
        if isinstance(velocity_y, torch.Tensor):
            velocity_y = velocity_y.detach().cpu().numpy()
        
        # Compute gradients
        dx = np.gradient(velocity_x, axis=1)
        dy = np.gradient(velocity_y, axis=0)
        
        # Divergence
        divergence = dx + dy
        
        return np.mean(divergence ** 2)
    
    @staticmethod
    def vorticity_error(velocity_x: Union[np.ndarray, torch.Tensor],
                       velocity_y: Union[np.ndarray, torch.Tensor]) -> float:
        """Compute vorticity error."""
        if isinstance(velocity_x, torch.Tensor):
            velocity_x = velocity_x.detach().cpu().numpy()
        if isinstance(velocity_y, torch.Tensor):
            velocity_y = velocity_y.detach().cpu().numpy()
        
        # Compute gradients
        dx = np.gradient(velocity_x, axis=1)
        dy = np.gradient(velocity_y, axis=0)
        
        # Vorticity
        vorticity = dx - dy
        
        return np.mean(vorticity ** 2)
    
    @staticmethod
    def energy_spectrum_error(pred: Union[np.ndarray, torch.Tensor],
                             target: Union[np.ndarray, torch.Tensor]) -> float:
        """Compute energy spectrum error."""
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        
        # Compute 2D FFT
        pred_fft = np.fft.fft2(pred)
        target_fft = np.fft.fft2(target)
        
        # Power spectrum
        pred_power = np.abs(pred_fft) ** 2
        target_power = np.abs(target_fft) ** 2
        
        # Error in power spectrum
        return np.mean((pred_power - target_power) ** 2)
    
    @staticmethod
    def compute_all_metrics(pred: Union[np.ndarray, torch.Tensor],
                           target: Union[np.ndarray, torch.Tensor],
                           velocity_fields: Optional[Dict[str, Union[np.ndarray, torch.Tensor]]] = None) -> Dict[str, float]:
        """
        Compute all available metrics.
        
        Args:
            pred: Predicted values
            target: Target values
            velocity_fields: Dictionary containing velocity components
            
        Returns:
            Dictionary containing all metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['mse'] = FluidMetrics.mse(pred, target)
        metrics['mae'] = FluidMetrics.mae(pred, target)
        metrics['rmse'] = FluidMetrics.rmse(pred, target)
        metrics['relative_error'] = FluidMetrics.relative_error(pred, target)
        metrics['correlation'] = FluidMetrics.correlation_coefficient(pred, target)
        
        # Physics-based metrics
        if velocity_fields and 'velocity_x' in velocity_fields and 'velocity_y' in velocity_fields:
            metrics['divergence_error'] = FluidMetrics.divergence_error(
                velocity_fields['velocity_x'], velocity_fields['velocity_y']
            )
            metrics['vorticity_error'] = FluidMetrics.vorticity_error(
                velocity_fields['velocity_x'], velocity_fields['velocity_y']
            )
        
        # Spectral metrics
        metrics['energy_spectrum_error'] = FluidMetrics.energy_spectrum_error(pred, target)
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Metrics"):
        """Print metrics in a formatted way."""
        print(f"\n{title}")
        print("=" * len(title))
        
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                print(f"{metric_name:25s}: {value:10.6f}")
            else:
                print(f"{metric_name:25s}: {value}")
    
    @staticmethod
    def compare_models(model_predictions: Dict[str, Union[np.ndarray, torch.Tensor]],
                       target: Union[np.ndarray, torch.Tensor],
                       velocity_fields: Optional[Dict[str, Union[np.ndarray, torch.Tensor]]] = None) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple models using the same metrics.
        
        Args:
            model_predictions: Dictionary of model predictions
            target: Target values
            velocity_fields: Velocity field components
            
        Returns:
            Dictionary containing metrics for each model
        """
        comparison = {}
        
        for model_name, pred in model_predictions.items():
            metrics = FluidMetrics.compute_all_metrics(pred, target, velocity_fields)
            comparison[model_name] = metrics
        
        return comparison
    
    @staticmethod
    def print_model_comparison(comparison: Dict[str, Dict[str, float]]):
        """Print model comparison in a formatted table."""
        if not comparison:
            return
        
        # Get all metric names
        all_metrics = set()
        for model_metrics in comparison.values():
            all_metrics.update(model_metrics.keys())
        
        all_metrics = sorted(list(all_metrics))
        
        # Print header
        print("\nModel Comparison")
        print("=" * 80)
        print(f"{'Model':<20s}", end="")
        for metric in all_metrics:
            print(f"{metric:>12s}", end="")
        print()
        print("-" * 80)
        
        # Print metrics for each model
        for model_name, metrics in comparison.items():
            print(f"{model_name:<20s}", end="")
            for metric in all_metrics:
                value = metrics.get(metric, "N/A")
                if isinstance(value, float):
                    print(f"{value:>12.6f}", end="")
                else:
                    print(f"{value:>12s}", end="")
            print()
        
        print("=" * 80)
    
    def __repr__(self) -> str:
        return "FluidMetrics()"
