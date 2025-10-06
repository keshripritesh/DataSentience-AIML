"""
Simulation Generator for FluidNetSim.

Generates training data sequences for neural network training.
"""

import numpy as np
import os
import h5py
from typing import Tuple, Dict, List, Optional, Any
from .core.fluid_solver import FluidSimulator
import logging
from tqdm import tqdm
import argparse

logger = logging.getLogger(__name__)

class SimulationGenerator:
    """
    Generates training data sequences for fluid dynamics simulation.
    
    Features:
    - Multiple simulation scenarios
    - Parameter variation
    - Data augmentation
    - Efficient storage formats
    """
    
    def __init__(
        self,
        output_dir: str = "training_data",
        resolution: Tuple[int, int] = (256, 256),
        physics_engine: str = "lattice_boltzmann",
        **kwargs
    ):
        """
        Initialize simulation generator.
        
        Args:
            output_dir: Directory to save training data
            resolution: Grid resolution for simulations
            physics_engine: Physics engine to use
            **kwargs: Additional simulator parameters
        """
        self.output_dir = output_dir
        self.resolution = resolution
        self.physics_engine = physics_engine
        self.simulator_kwargs = kwargs
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Simulation scenarios
        self.scenarios = self._define_scenarios()
        
        logger.info(f"Initialized simulation generator: {resolution}, engine={physics_engine}")
    
    def _define_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """Define different simulation scenarios."""
        return {
            "laminar_flow": {
                "description": "Laminar flow in a channel",
                "parameters": {
                    "viscosity": 0.01,
                    "inflow_velocity": 0.1,
                    "obstacles": False
                }
            },
            "turbulent_flow": {
                "description": "Turbulent flow with obstacles",
                "parameters": {
                    "viscosity": 0.001,
                    "inflow_velocity": 0.5,
                    "obstacles": True
                }
            },
            "vortex_shedding": {
                "description": "Vortex shedding behind cylinder",
                "parameters": {
                    "viscosity": 0.005,
                    "inflow_velocity": 0.3,
                    "obstacles": True,
                    "obstacle_type": "cylinder"
                }
            },
            "mixing_layer": {
                "description": "Mixing layer between two streams",
                "parameters": {
                    "viscosity": 0.002,
                    "inflow_velocity": 0.2,
                    "obstacles": False,
                    "multi_stream": True
                }
            }
        }
    
    def generate_sequence(
        self,
        scenario: str,
        num_steps: int = 100,
        dt: float = 0.01,
        save_frames: bool = True,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Generate a single simulation sequence.
        
        Args:
            scenario: Simulation scenario name
            num_steps: Number of timesteps
            dt: Timestep size
            save_frames: Whether to save individual frames
            **kwargs: Additional scenario parameters
            
        Returns:
            Dictionary containing the sequence data
        """
        if scenario not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario}")
        
        # Get scenario parameters
        params = self.scenarios[scenario]["parameters"].copy()
        params.update(kwargs)
        
        # Initialize simulator
        simulator = FluidSimulator(
            resolution=self.resolution,
            physics_engine=self.physics_engine,
            **self.simulator_kwargs
        )
        
        # Setup scenario-specific configuration
        self._setup_scenario(simulator, scenario, params)
        
        # Run simulation
        logger.info(f"Generating {scenario} sequence: {num_steps} steps")
        
        sequence_data = []
        for step in tqdm(range(num_steps), desc=f"Generating {scenario}"):
            state = simulator.step(dt)
            
            if save_frames:
                # Extract relevant fields for training
                frame_data = self._extract_frame_data(state, scenario)
                sequence_data.append(frame_data)
        
        # Convert to numpy arrays
        sequence = {
            "scenario": scenario,
            "parameters": params,
            "timesteps": num_steps,
            "dt": dt,
            "resolution": self.resolution
        }
        
        for key in sequence_data[0].keys():
            sequence[key] = np.array([frame[key] for frame in sequence_data])
        
        return sequence
    
    def _setup_scenario(self, simulator: FluidSimulator, scenario: str, params: Dict[str, Any]):
        """Setup simulator for specific scenario."""
        if params.get("obstacles", False):
            self._add_obstacles(simulator, params)
        
        if params.get("multi_stream", False):
            self._setup_multi_stream(simulator, params)
        
        # Set boundary conditions
        simulator.set_boundary_conditions("periodic")
    
    def _add_obstacles(self, simulator: FluidSimulator, params: Dict[str, Any]):
        """Add obstacles to the simulation domain."""
        obstacle_type = params.get("obstacle_type", "rectangle")
        
        if obstacle_type == "cylinder":
            # Add circular obstacle
            center_x, center_y = self.resolution[0] // 3, self.resolution[1] // 2
            radius = min(self.resolution) // 10
            
            # Create circular mask
            x, y = np.ogrid[:self.resolution[0], :self.resolution[1]]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            
            simulator.add_obstacle(mask, (0, 0))
        
        elif obstacle_type == "rectangle":
            # Add rectangular obstacle
            width, height = self.resolution[0] // 8, self.resolution[1] // 4
            x, y = self.resolution[0] // 3, self.resolution[1] // 3
            
            mask = np.ones((width, height), dtype=bool)
            simulator.add_obstacle(mask, (x, y))
    
    def _setup_multi_stream(self, simulator: FluidSimulator, params: Dict[str, Any]):
        """Setup multi-stream inflow conditions."""
        # This would implement multi-stream inflow
        # For now, just a placeholder
        pass
    
    def _extract_frame_data(self, state: Dict[str, np.ndarray], scenario: str) -> Dict[str, np.ndarray]:
        """Extract relevant data from simulation state for training."""
        frame_data = {}
        
        if "velocity_x" in state and "velocity_y" in state:
            # Velocity magnitude
            frame_data["velocity_magnitude"] = np.sqrt(
                state["velocity_x"]**2 + state["velocity_y"]**2
            )
            
            # Velocity components
            frame_data["velocity_x"] = state["velocity_x"]
            frame_data["velocity_y"] = state["velocity_y"]
        
        if "density" in state:
            frame_data["density"] = state["density"]
        
        if "pressure" in state:
            frame_data["pressure"] = state["pressure"]
        
        # Add boundary mask if available
        if "boundary_mask" in state:
            frame_data["boundary_mask"] = state["boundary_mask"]
        
        return frame_data
    
    def generate_dataset(
        self,
        num_sequences: int = 1000,
        timesteps: int = 50,
        scenarios: Optional[List[str]] = None,
        **kwargs
    ) -> None:
        """
        Generate a complete training dataset.
        
        Args:
            num_sequences: Total number of sequences to generate
            timesteps: Number of timesteps per sequence
            scenarios: List of scenarios to use (None for all)
            **kwargs: Additional parameters
        """
        if scenarios is None:
            scenarios = list(self.scenarios.keys())
        
        sequences_per_scenario = num_sequences // len(scenarios)
        
        logger.info(f"Generating dataset: {num_sequences} sequences, {timesteps} timesteps")
        
        for scenario in scenarios:
            logger.info(f"Generating {sequences_per_scenario} sequences for {scenario}")
            
            for i in range(sequences_per_scenario):
                # Add some parameter variation
                params = self._vary_parameters(scenario, i)
                
                try:
                    sequence = self.generate_sequence(
                        scenario=scenario,
                        num_steps=timesteps,
                        **params
                    )
                    
                    # Save sequence
                    filename = f"{scenario}_seq_{i:04d}.h5"
                    filepath = os.path.join(self.output_dir, filename)
                    self._save_sequence(sequence, filepath)
                    
                except Exception as e:
                    logger.error(f"Failed to generate sequence {i} for {scenario}: {e}")
                    continue
        
        logger.info(f"Dataset generation complete: {self.output_dir}")
    
    def _vary_parameters(self, scenario: str, sequence_idx: int) -> Dict[str, Any]:
        """Vary parameters for data augmentation."""
        base_params = self.scenarios[scenario]["parameters"].copy()
        
        # Add random variations
        np.random.seed(sequence_idx)  # For reproducibility
        
        if "viscosity" in base_params:
            base_params["viscosity"] *= np.random.uniform(0.8, 1.2)
        
        if "inflow_velocity" in base_params:
            base_params["inflow_velocity"] *= np.random.uniform(0.9, 1.1)
        
        return base_params
    
    def _save_sequence(self, sequence: Dict[str, np.ndarray], filepath: str):
        """Save sequence data to HDF5 file."""
        with h5py.File(filepath, 'w') as f:
            # Save metadata
            for key, value in sequence.items():
                if isinstance(value, (str, int, float, tuple)):
                    f.attrs[key] = value
            
            # Save arrays
            for key, value in sequence.items():
                if isinstance(value, np.ndarray):
                    f.create_dataset(key, data=value, compression="gzip", compression_opts=9)
    
    def load_sequence(self, filepath: str) -> Dict[str, np.ndarray]:
        """Load sequence data from HDF5 file."""
        with h5py.File(filepath, 'r') as f:
            sequence = {}
            
            # Load metadata
            for key, value in f.attrs.items():
                sequence[key] = value
            
            # Load arrays
            for key in f.keys():
                sequence[key] = f[key][:]
        
        return sequence
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the generated dataset."""
        if not os.path.exists(self.output_dir):
            return {"status": "No dataset generated yet"}
        
        files = [f for f in os.listdir(self.output_dir) if f.endswith('.h5')]
        
        info = {
            "output_directory": self.output_dir,
            "total_files": len(files),
            "file_size_total": sum(os.path.getsize(os.path.join(self.output_dir, f)) for f in files),
            "scenarios": list(set(f.split('_')[0] for f in files))
        }
        
        return info

def main():
    """Command-line interface for simulation generator."""
    parser = argparse.ArgumentParser(description="Generate training data for FluidNetSim")
    parser.add_argument("--num_sequences", type=int, default=100, help="Number of sequences to generate")
    parser.add_argument("--timesteps", type=int, default=50, help="Timesteps per sequence")
    parser.add_argument("--resolution", type=str, default="256x256", help="Grid resolution")
    parser.add_argument("--output_dir", type=str, default="training_data", help="Output directory")
    parser.add_argument("--scenarios", type=str, nargs="+", help="Specific scenarios to generate")
    
    args = parser.parse_args()
    
    # Parse resolution
    resolution = tuple(map(int, args.resolution.split('x')))
    
    # Initialize generator
    generator = SimulationGenerator(
        output_dir=args.output_dir,
        resolution=resolution
    )
    
    # Generate dataset
    generator.generate_dataset(
        num_sequences=args.num_sequences,
        timesteps=args.timesteps,
        scenarios=args.scenarios
    )
    
    # Print dataset info
    info = generator.get_dataset_info()
    print(f"Dataset generated successfully:")
    for key, value in info.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
