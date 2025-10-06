"""
Core simulation components for FluidNetSim.

Contains the fundamental fluid dynamics solvers and utilities.
"""

from .fluid_solver import FluidSimulator
from .lattice_boltzmann import LatticeBoltzmannSolver
from .navier_stokes import NavierStokesSolver
from .boundary_conditions import BoundaryConditions

__all__ = [
    "FluidSimulator",
    "LatticeBoltzmannSolver",
    "NavierStokesSolver", 
    "BoundaryConditions",
]
