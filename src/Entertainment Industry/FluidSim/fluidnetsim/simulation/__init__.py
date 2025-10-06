"""
Simulation module for FluidNetSim.

Provides advanced fluid dynamics simulation capabilities including:
- Lattice Boltzmann Method (LBM)
- Navier-Stokes solvers
- Multi-phase flow simulation
- Turbulence modeling
"""

from .core.fluid_solver import FluidSimulator
from .core.lattice_boltzmann import LatticeBoltzmannSolver
from .core.navier_stokes import NavierStokesSolver
from .generator import SimulationGenerator

__all__ = [
    "FluidSimulator",
    "LatticeBoltzmannSolver", 
    "NavierStokesSolver",
    "SimulationGenerator",
]
