"""
Machine Learning Module for FluidNetSim.

Provides neural network architectures and training capabilities for fluid dynamics prediction.
"""

from .models.convlstm_unet import ConvLSTMUNet
from .models.physics_informed import PhysicsInformedNet
from .training.trainer import FluidNetTrainer
from .evaluation.metrics import FluidMetrics

__all__ = [
    "ConvLSTMUNet",
    "PhysicsInformedNet", 
    "FluidNetTrainer",
    "FluidMetrics",
]
