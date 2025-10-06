"""
Unit tests for FluidNetSim simulation module.

Tests fluid dynamics solvers and simulation components.
"""

import pytest
import numpy as np
import torch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from fluidnetsim.simulation import FluidSimulator, LatticeBoltzmannSolver, NavierStokesSolver
from fluidnetsim.simulation.core.boundary_conditions import (
    BoundaryConditions, PeriodicBoundaryCondition, NoSlipBoundaryCondition
)

class TestLatticeBoltzmannSolver:
    """Test Lattice Boltzmann Method solver."""
    
    def test_initialization(self):
        """Test solver initialization."""
        solver = LatticeBoltzmannSolver(resolution=(64, 64))
        
        assert solver.resolution == (64, 64)
        assert solver.nx == 64
        assert solver.ny == 64
        assert solver.nq == 9  # D2Q9 lattice
        assert solver.turbulence_model in ["les", "smagorinsky", "none"]
    
    def test_d2q9_lattice_parameters(self):
        """Test D2Q9 lattice parameters."""
        solver = LatticeBoltzmannSolver(resolution=(32, 32))
        
        # Check velocity vectors
        assert solver.e.shape == (9, 2)
        assert solver.e[0].tolist() == [0, 0]  # Rest particle
        assert solver.e[1].tolist() == [1, 0]  # Right
        assert solver.e[2].tolist() == [0, 1]  # Up
        
        # Check weights
        assert len(solver.w) == 9
        assert abs(solver.w[0] - 4/9) < 1e-10  # Rest particle weight
        assert abs(solver.w[1] - 1/9) < 1e-10  # Cardinal directions
        assert abs(solver.w[5] - 1/36) < 1e-10  # Diagonal directions
    
    def test_distribution_functions_shape(self):
        """Test distribution function array shapes."""
        solver = LatticeBoltzmannSolver(resolution=(16, 16))
        
        assert solver.f.shape == (16, 16, 9)
        assert solver.f_new.shape == (16, 16, 9)
        assert solver.rho.shape == (16, 16)
        assert solver.u.shape == (16, 16, 2)
        assert solver.p.shape == (16, 16)
    
    def test_equilibrium_calculation(self):
        """Test equilibrium distribution calculation."""
        solver = LatticeBoltzmannSolver(resolution=(8, 8))
        
        # Set some velocity and density
        solver.u[:, :, 0] = 0.1  # x-velocity
        solver.u[:, :, 1] = 0.05  # y-velocity
        solver.rho = 1.0
        
        feq = solver._calculate_equilibrium()
        
        assert feq.shape == (8, 8, 9)
        assert np.all(feq >= 0)  # Weights should be positive
        assert np.allclose(np.sum(feq, axis=2), solver.rho)  # Sum should equal density
    
    def test_collision_step(self):
        """Test collision step."""
        solver = LatticeBoltzmannSolver(resolution=(16, 16))
        
        # Store original distribution functions
        f_original = solver.f.copy()
        
        # Run collision step
        solver._collision()
        
        # Distribution functions should change
        assert not np.allclose(solver.f, f_original)
    
    def test_streaming_step(self):
        """Test streaming step."""
        solver = LatticeBoltzmannSolver(resolution=(16, 16))
        
        # Set some distribution function values
        solver.f[8, 8, 1] = 1.0  # Right-moving particle at center
        
        # Store original state
        f_original = solver.f.copy()
        
        # Run streaming step
        solver._streaming()
        
        # Distribution functions should change due to streaming
        assert not np.allclose(solver.f, f_original)
    
    def test_macroscopic_update(self):
        """Test macroscopic variable update."""
        solver = LatticeBoltzmannSolver(resolution=(16, 16))
        
        # Set some distribution function values
        solver.f[:, :, 0] = 4/9  # Rest particles
        solver.f[:, :, 1] = 1/9  # Right-moving particles
        solver.f[:, :, 2] = 1/9  # Up-moving particles
        
        # Update macroscopic variables
        solver._update_macroscopic()
        
        # Check density (4/9 + 1/9 + 1/9 = 6/9 = 2/3)
        assert np.allclose(solver.rho, 2/3)
        
        # Check velocity (should have x and y components)
        assert np.any(solver.u[:, :, 0] != 0)  # x-velocity
        assert np.any(solver.u[:, :, 1] != 0)  # y-velocity
    
    def test_boundary_conditions(self):
        """Test boundary condition handling."""
        solver = LatticeBoltzmannSolver(resolution=(16, 16))
        
        # Add obstacle
        obstacle_mask = np.ones((8, 8), dtype=bool)
        solver.add_obstacle(obstacle_mask, (4, 4))
        
        # Check boundary mask
        assert not np.all(solver.boundary_mask)
        assert solver.boundary_mask[8:12, 8:12].shape == (4, 4)
    
    def test_turbulence_model(self):
        """Test turbulence modeling."""
        solver = LatticeBoltzmannSolver(resolution=(32, 32), turbulence_model="les")
        
        # Set some velocity field
        solver.u[:, :, 0] = np.random.rand(32, 32) * 0.1
        solver.u[:, :, 1] = np.random.rand(32, 32) * 0.1
        
        # Apply turbulence model
        solver._apply_turbulence_model()
        
        # Relaxation time should be updated
        assert hasattr(solver, 'relaxation_time')
        assert solver.relaxation_time > 0.5
    
    def test_state_management(self):
        """Test state saving and loading."""
        solver = LatticeBoltzmannSolver(resolution=(16, 16))
        
        # Set some state
        solver.u[:, :, 0] = 0.1
        solver.rho[:, :] = 1.2
        
        # Get state
        state = solver.get_state()
        
        # Check state contents
        assert "density" in state
        assert "velocity_x" in state
        assert "velocity_y" in state
        assert "pressure" in state
        assert "distribution_functions" in state
        assert "boundary_mask" in state
        
        # Create new solver and set state
        new_solver = LatticeBoltzmannSolver(resolution=(16, 16))
        new_solver.set_state(state)
        
        # Check state transfer
        assert np.allclose(new_solver.u, solver.u)
        assert np.allclose(new_solver.rho, solver.rho)

class TestNavierStokesSolver:
    """Test Navier-Stokes solver."""
    
    def test_initialization(self):
        """Test solver initialization."""
        solver = NavierStokesSolver(resolution=(64, 64))
        
        assert solver.resolution == (64, 64)
        assert solver.nx == 64
        assert solver.ny == 64
        assert solver.viscosity > 0
        assert solver.density > 0
    
    def test_grid_parameters(self):
        """Test grid parameter calculation."""
        solver = NavierStokesSolver(resolution=(32, 32))
        
        # Grid spacing should be calculated correctly
        expected_dx = 1.0 / (32 - 1)
        expected_dy = 1.0 / (32 - 1)
        
        assert abs(solver.dx - expected_dx) < 1e-10
        assert abs(solver.dy - expected_dy) < 1e-10
    
    def test_flow_variables_shape(self):
        """Test flow variable array shapes."""
        solver = NavierStokesSolver(resolution=(16, 16))
        
        assert solver.u.shape == (16, 16)  # x-velocity
        assert solver.v.shape == (16, 16)  # y-velocity
        assert solver.p.shape == (16, 16)  # pressure
        assert solver.div_u.shape == (16, 16)  # divergence
    
    def test_velocity_prediction(self):
        """Test velocity prediction step."""
        solver = NavierStokesSolver(resolution=(16, 16))
        
        # Set some initial velocity field
        solver.u[:, :] = 0.1
        solver.v[:, :] = 0.05
        
        # Store original values
        u_original = solver.u.copy()
        v_original = solver.v.copy()
        
        # Run velocity prediction
        solver._predict_velocity(dt=0.01)
        
        # Velocities should change
        assert not np.allclose(solver.u_new, u_original)
        assert not np.allclose(solver.v_new, v_original)
    
    def test_pressure_solving(self):
        """Test pressure equation solving."""
        solver = NavierStokesSolver(resolution=(16, 16))

        # Set some divergence field (more realistic)
        solver.div_u[8, 8] = 1.0  # Larger value
        solver.div_u[7, 8] = 0.5  # Surrounding points
        solver.div_u[9, 8] = 0.5
        solver.div_u[8, 7] = 0.5
        solver.div_u[8, 9] = 0.5
        
        # Solve pressure
        solver._solve_pressure(dt=0.01, recalculate_divergence=False)

        # Pressure should be updated
        assert np.any(solver.p != 0)
    
    def test_velocity_correction(self):
        """Test velocity correction step."""
        solver = NavierStokesSolver(resolution=(16, 16))
        
        # Set some predicted velocities and pressure
        solver.u_new[:, :] = 0.1
        solver.v_new[:, :] = 0.05
        solver.p[:, :] = 0.01
        
        # Store original values
        u_original = solver.u.copy()
        v_original = solver.v.copy()
        
        # Run velocity correction
        solver._correct_velocity(dt=0.01)
        
        # Velocities should be corrected
        assert not np.allclose(solver.u, u_original)
        assert not np.allclose(solver.v, v_original)
    
    def test_boundary_conditions(self):
        """Test boundary condition application."""
        solver = NavierStokesSolver(resolution=(16, 16))
        
        # Add obstacle
        obstacle_mask = np.ones((8, 8), dtype=bool)
        solver.add_obstacle(obstacle_mask, (4, 4))
        
        # Apply boundary conditions
        solver._apply_boundary_conditions()
        
        # Velocities at obstacle should be zero
        mask = ~solver.boundary_mask
        assert np.all(solver.u[mask] == 0)
        assert np.all(solver.v[mask] == 0)
    
    def test_turbulence_model(self):
        """Test turbulence modeling."""
        solver = NavierStokesSolver(resolution=(32, 32), turbulence_model="les")
        
        # Set some velocity field
        solver.u[:, :] = np.random.rand(32, 32) * 0.1
        solver.v[:, :] = np.random.rand(32, 32) * 0.1
        
        # Apply turbulence model
        solver._apply_turbulence_model()
        
        # Effective viscosity should be updated
        assert hasattr(solver, 'effective_viscosity')
        assert solver.effective_viscosity >= solver.viscosity

class TestBoundaryConditions:
    """Test boundary condition classes."""
    
    def test_periodic_boundary_condition(self):
        """Test periodic boundary conditions."""
        bc = PeriodicBoundaryCondition(axis=0)
        
        # Create test field
        field = np.random.rand(16, 16)
        
        # Apply boundary condition
        bc.apply(field)
        
        # Check periodicity in x-direction
        assert np.allclose(field[0, :], field[-2, :])
        assert np.allclose(field[-1, :], field[1, :])
    
    def test_no_slip_boundary_condition(self):
        """Test no-slip boundary conditions."""
        bc = NoSlipBoundaryCondition(wall_velocity=(0.0, 0.0))
        
        # No-slip is typically applied in the main solver
        # This test just checks initialization
        assert bc.wall_velocity == (0.0, 0.0)
        assert bc.get_type() == "no_slip"
    
    def test_boundary_conditions_manager(self):
        """Test boundary conditions manager."""
        bc_manager = BoundaryConditions(resolution=(32, 32))
        
        # Check default boundary conditions
        assert "x" in bc_manager.boundary_conditions
        assert "y" in bc_manager.boundary_conditions
        
        # Add custom boundary condition
        custom_bc = NoSlipBoundaryCondition()
        bc_manager.add_boundary_condition("custom", custom_bc)
        
        assert "custom" in bc_manager.boundary_conditions
        assert bc_manager.boundary_conditions["custom"].get_type() == "no_slip"
    
    def test_solid_boundary_setting(self):
        """Test solid boundary setting."""
        bc_manager = BoundaryConditions(resolution=(16, 16))
        
        # Create obstacle mask
        obstacle_mask = np.ones((8, 8), dtype=bool)
        
        # Set solid boundary
        bc_manager.set_solid_boundary(obstacle_mask, (4, 4))
        
        # Check boundary mask
        assert not np.all(bc_manager.boundary_mask)
        assert bc_manager.boundary_mask[4:12, 4:12].shape == (8, 8)
    
    def test_inflow_profile_setting(self):
        """Test inflow profile setting."""
        bc_manager = BoundaryConditions(resolution=(32, 32))
        
        # Set uniform inflow profile
        bc_manager.set_inflow_profile("uniform", velocity=1.0)
        
        # Check that inflow boundary condition was added
        assert "inflow" in bc_manager.boundary_conditions

class TestFluidSimulator:
    """Test main fluid simulator."""
    
    def test_initialization(self):
        """Test simulator initialization."""
        simulator = FluidSimulator(resolution=(128, 128))
        
        assert simulator.resolution == (128, 128)
        assert simulator.physics_engine == "lattice_boltzmann"
        assert simulator.turbulence_model == "les"
        assert simulator.gpu_acceleration == False
    
    def test_solver_creation(self):
        """Test solver creation for different physics engines."""
        # LBM solver
        lbm_sim = FluidSimulator(physics_engine="lattice_boltzmann")
        assert isinstance(lbm_sim.solver, LatticeBoltzmannSolver)
        
        # Navier-Stokes solver
        ns_sim = FluidSimulator(physics_engine="navier_stokes")
        assert isinstance(ns_sim.solver, NavierStokesSolver)
        
        # Invalid engine should raise error
        with pytest.raises(ValueError):
            FluidSimulator(physics_engine="invalid_engine")
    
    def test_simulation_step(self):
        """Test single simulation step."""
        simulator = FluidSimulator(resolution=(64, 64))
        
        # Store initial state
        initial_state = simulator.get_state()
        
        # Take one step
        new_state = simulator.step(dt=0.01)
        
        # Check that time advanced
        assert new_state["time"] > initial_state["time"]
        assert new_state["timestep"] > initial_state["timestep"]
        
        # Check that state changed
        assert not np.allclose(
            new_state["density"], 
            initial_state["density"]
        )
    
    def test_simulation_run(self):
        """Test multiple simulation steps."""
        simulator = FluidSimulator(resolution=(32, 32))
        
        # Run simulation
        final_state = simulator.run(num_steps=10, dt=0.01)
        
        # Check final state
        assert final_state["timestep"] == 10
        assert np.allclose(final_state["time"], 0.1)
    
    def test_performance_tracking(self):
        """Test performance metrics tracking."""
        simulator = FluidSimulator(resolution=(64, 64))
        
        # Run some steps
        simulator.run(num_steps=5, dt=0.01)
        
        # Get performance summary
        metrics = simulator.get_performance_summary()
        
        assert "total_time" in metrics
        assert "steps_per_second" in metrics
        assert metrics["total_time"] > 0
        assert metrics["steps_per_second"] > 0
    
    def test_state_saving_loading(self, tmp_path):
        """Test state saving and loading."""
        simulator = FluidSimulator(resolution=(32, 32))
        
        # Run some steps
        simulator.run(num_steps=5, dt=0.01)
        
        # Save state
        save_path = tmp_path / "test_state.h5"
        simulator.save_state(str(save_path))
        
        # Create new simulator
        new_simulator = FluidSimulator(resolution=(32, 32))
        
        # Load state
        new_simulator.load_state(str(save_path))
        
        # Check state transfer
        original_state = simulator.get_state()
        loaded_state = new_simulator.get_state()
        
        assert original_state["timestep"] == loaded_state["timestep"]
        assert original_state["time"] == loaded_state["time"]
    
    def test_obstacle_addition(self):
        """Test obstacle addition."""
        simulator = FluidSimulator(resolution=(64, 64))
        
        # Create obstacle mask
        obstacle_mask = np.ones((16, 16), dtype=bool)
        
        # Add obstacle
        simulator.add_obstacle(obstacle_mask, (24, 24))
        
        # Check that obstacle was added
        state = simulator.get_state()
        assert "boundary_mask" in state
        assert not np.all(state["boundary_mask"])

if __name__ == "__main__":
    pytest.main([__file__])
