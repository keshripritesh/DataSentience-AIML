"""
Boundary Conditions Module for FluidNetSim.

Implements various boundary condition types for fluid dynamics simulations.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, Callable
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class BoundaryCondition(ABC):
    """Abstract base class for boundary conditions."""
    
    @abstractmethod
    def apply(self, field: np.ndarray, **kwargs) -> None:
        """Apply boundary condition to the field."""
        pass
    
    @abstractmethod
    def get_type(self) -> str:
        """Get boundary condition type."""
        pass

class PeriodicBoundaryCondition(BoundaryCondition):
    """Periodic boundary conditions for periodic domains."""
    
    def __init__(self, axis: int = 0):
        """
        Initialize periodic boundary condition.
        
        Args:
            axis: Axis along which to apply periodicity (0 for x, 1 for y)
        """
        self.axis = axis
    
    def apply(self, field: np.ndarray, **kwargs) -> None:
        """Apply periodic boundary condition."""
        if self.axis == 0:  # X-direction
            field[0, :] = field[-2, :]
            field[-1, :] = field[1, :]
        elif self.axis == 1:  # Y-direction
            field[:, 0] = field[:, -2]
            field[:, -1] = field[:, 1]
    
    def get_type(self) -> str:
        return "periodic"

class NoSlipBoundaryCondition(BoundaryCondition):
    """No-slip boundary condition for solid walls."""
    
    def __init__(self, wall_velocity: Tuple[float, float] = (0.0, 0.0)):
        """
        Initialize no-slip boundary condition.
        
        Args:
            wall_velocity: Velocity of the wall (u, v)
        """
        self.wall_velocity = wall_velocity
    
    def apply(self, field: np.ndarray, **kwargs) -> None:
        """Apply no-slip boundary condition."""
        # This is typically applied in the main solver
        # by setting velocity to wall velocity at boundary
        pass
    
    def get_type(self) -> str:
        return "no_slip"

class SlipBoundaryCondition(BoundaryCondition):
    """Slip boundary condition for free surfaces."""
    
    def __init__(self):
        """Initialize slip boundary condition."""
        pass
    
    def apply(self, field: np.ndarray, **kwargs) -> None:
        """Apply slip boundary condition."""
        # Normal velocity component is zero
        # Tangential velocity component is extrapolated
        pass
    
    def get_type(self) -> str:
        return "slip"

class InflowBoundaryCondition(BoundaryCondition):
    """Inflow boundary condition with specified velocity profile."""
    
    def __init__(self, velocity_profile: Callable, axis: int = 0):
        """
        Initialize inflow boundary condition.
        
        Args:
            velocity_profile: Function that returns velocity profile
            axis: Axis along which inflow occurs
        """
        self.velocity_profile = velocity_profile
        self.axis = axis
    
    def apply(self, field: np.ndarray, **kwargs) -> None:
        """Apply inflow boundary condition."""
        if self.axis == 0:  # X-direction
            field[0, :] = self.velocity_profile(field.shape[1], **kwargs)
        elif self.axis == 1:  # Y-direction
            field[:, 0] = self.velocity_profile(field.shape[0], **kwargs)
    
    def get_type(self) -> str:
        return "inflow"

class OutflowBoundaryCondition(BoundaryCondition):
    """Outflow boundary condition with zero gradient."""
    
    def __init__(self, axis: int = 0):
        """
        Initialize outflow boundary condition.
        
        Args:
            axis: Axis along which outflow occurs
        """
        self.axis = axis
    
    def apply(self, field: np.ndarray, **kwargs) -> None:
        """Apply outflow boundary condition."""
        if self.axis == 0:  # X-direction
            field[-1, :] = field[-2, :]  # Zero gradient
        elif self.axis == 1:  # Y-direction
            field[:, -1] = field[:, -2]  # Zero gradient
    
    def get_type(self) -> str:
        return "outflow"

class BoundaryConditions:
    """Manager class for multiple boundary conditions."""
    
    def __init__(self, resolution: Tuple[int, int]):
        """
        Initialize boundary conditions manager.
        
        Args:
            resolution: Grid resolution (width, height)
        """
        self.resolution = resolution
        self.nx, self.ny = resolution
        self.boundary_conditions = {}
        self.boundary_mask = np.ones((self.nx, self.ny), dtype=bool)
        
        # Default boundary conditions
        self._setup_default_boundaries()
    
    def _setup_default_boundaries(self):
        """Setup default boundary conditions."""
        # Periodic in both directions by default
        self.add_boundary_condition("x", PeriodicBoundaryCondition(axis=0))
        self.add_boundary_condition("y", PeriodicBoundaryCondition(axis=1))
    
    def add_boundary_condition(self, boundary: str, condition: BoundaryCondition):
        """
        Add boundary condition for a specific boundary.
        
        Args:
            boundary: Boundary identifier ("x", "y", "left", "right", "top", "bottom")
            condition: Boundary condition object
        """
        self.boundary_conditions[boundary] = condition
        logger.info(f"Added {condition.get_type()} boundary condition for {boundary}")
    
    def apply_boundary_conditions(self, field: np.ndarray, field_type: str = "velocity"):
        """
        Apply all boundary conditions to a field.
        
        Args:
            field: Field to apply boundary conditions to
            field_type: Type of field ("velocity", "pressure", "density")
        """
        for boundary, condition in self.boundary_conditions.items():
            condition.apply(field, field_type=field_type)
    
    def set_solid_boundary(self, mask: np.ndarray, position: Tuple[int, int]):
        """
        Set solid boundary using a mask.
        
        Args:
            mask: Boolean mask indicating solid regions
            position: Position to place the mask
        """
        x, y = position
        h, w = mask.shape
        
        # Ensure mask fits within domain
        x_end = min(x + w, self.nx)
        y_end = min(y + h, self.ny)
        x_start = max(x, 0)
        y_start = max(y, 0)
        
        # Update boundary mask
        self.boundary_mask[x_start:x_end, y_start:y_end] = ~mask[
            x_start-x:x_end-x, y_start-y:y_end-y
        ]
    
    def get_boundary_mask(self) -> np.ndarray:
        """Get the boundary mask."""
        return self.boundary_mask.copy()
    
    def set_inflow_profile(self, profile: str, **kwargs):
        """
        Set inflow velocity profile.
        
        Args:
            profile: Profile type ("uniform", "parabolic", "gaussian")
            **kwargs: Profile-specific parameters
        """
        if profile == "uniform":
            velocity_func = lambda n, **kw: np.full(n, kw.get("velocity", 1.0))
        elif profile == "parabolic":
            velocity_func = lambda n, **kw: kw.get("velocity", 1.0) * (1 - (np.arange(n) - n/2)**2 / (n/2)**2)
        elif profile == "gaussian":
            velocity_func = lambda n, **kw: kw.get("velocity", 1.0) * np.exp(-0.5 * ((np.arange(n) - n/2) / (n/4))**2)
        else:
            raise ValueError(f"Unknown profile type: {profile}")
        
        # Add inflow boundary condition
        self.add_boundary_condition("inflow", InflowBoundaryCondition(velocity_func))
    
    def __repr__(self) -> str:
        return f"BoundaryConditions(resolution={self.resolution}, types={list(self.boundary_conditions.keys())})"
