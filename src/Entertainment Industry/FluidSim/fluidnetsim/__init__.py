"""
FluidNetSim: Advanced Physics-Informed Neural Fluid Dynamics Simulator

A breakthrough in computational fluid dynamics combining state-of-the-art 
physics-based simulation with cutting-edge deep learning techniques.
"""

__version__ = "1.0.0"
__author__ = "FluidNetSim Team"
__email__ = "contact@fluidnetsim.org"

# Core imports
from .simulation import FluidSimulator
from .ml import ConvLSTMUNet, PhysicsInformedNet
from .visualization import RealTimeVisualizer

# Version info
__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "FluidSimulator",
    "ConvLSTMUNet", 
    "PhysicsInformedNet",
    "RealTimeVisualizer",
]
