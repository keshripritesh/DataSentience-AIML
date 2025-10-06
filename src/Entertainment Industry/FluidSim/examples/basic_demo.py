#!/usr/bin/env python3
"""
Basic demonstration of FluidNetSim capabilities.

This script shows how to:
1. Initialize a fluid simulator
2. Run a simple simulation
3. Create and test neural network models
4. Generate training data
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fluidnetsim.simulation import FluidSimulator
from fluidnetsim.ml.models import ConvLSTMUNet, PhysicsInformedNet
from fluidnetsim.simulation.generator import SimulationGenerator

def demo_fluid_simulation():
    """Demonstrate basic fluid simulation."""
    print("🌊 Running Fluid Simulation Demo...")
    
    # Initialize simulator
    simulator = FluidSimulator(
        resolution=(64, 64),
        physics_engine="lattice_boltzmann",
        turbulence_model="les"
    )
    
    print(f"✓ Initialized {simulator.physics_engine} simulator")
    print(f"  Resolution: {simulator.resolution}")
    print(f"  Turbulence model: {simulator.turbulence_model}")
    
    # Run simulation
    print("\n🔄 Running simulation...")
    final_state = simulator.run(num_steps=50, dt=0.01)
    
    print(f"✓ Simulation completed")
    print(f"  Final time: {final_state['time']:.3f}")
    print(f"  Timesteps: {final_state['timestep']}")
    
    # Get performance metrics
    metrics = simulator.get_performance_summary()
    print(f"  Performance: {metrics['steps_per_second']:.1f} steps/sec")
    
    return simulator, final_state

def demo_neural_networks():
    """Demonstrate neural network models."""
    print("\n🧠 Testing Neural Network Models...")
    
    # Test ConvLSTM-UNet
    print("  Testing ConvLSTM-UNet...")
    convlstm_model = ConvLSTMUNet(
        input_channels=3,
        hidden_channels=32,
        num_layers=3,
        output_channels=3
    )
    
    # Test forward pass
    import torch
    x = torch.randn(2, 3, 32, 32)
    output, hidden_states = convlstm_model(x)
    
    print(f"    ✓ ConvLSTM-UNet forward pass successful")
    print(f"      Input shape: {x.shape}")
    print(f"      Output shape: {output.shape}")
    print(f"      Hidden states: {len(hidden_states)} layers")
    
    # Test PINN
    print("  Testing Physics-Informed Neural Network...")
    pinn_model = PhysicsInformedNet(
        input_dim=4,
        hidden_dim=64,
        num_layers=4,
        output_dim=3
    )
    
    # Test forward pass
    x = torch.randn(2, 4)
    output = pinn_model(x)
    
    print(f"    ✓ PINN forward pass successful")
    print(f"      Input shape: {x.shape}")
    print(f"      Output shape: {output.shape}")
    
    # Get model information
    convlstm_info = convlstm_model.get_model_info()
    pinn_info = pinn_model.get_model_info()
    
    print(f"    ConvLSTM-UNet parameters: {convlstm_info['total_parameters']:,}")
    print(f"    PINN parameters: {pinn_info['total_parameters']:,}")
    
    return convlstm_model, pinn_model

def demo_training_data_generation():
    """Demonstrate training data generation."""
    print("\n📊 Testing Training Data Generation...")
    
    # Initialize generator
    generator = SimulationGenerator(
        output_dir="demo_training_data",
        resolution=(32, 32),
        physics_engine="lattice_boltzmann"
    )
    
    print(f"✓ Initialized simulation generator")
    print(f"  Output directory: {generator.output_dir}")
    print(f"  Resolution: {generator.resolution}")
    
    # Generate a single sequence
    print("  Generating sample sequence...")
    sequence = generator.generate_sequence(
        scenario="laminar_flow",
        num_steps=20,
        dt=0.01
    )
    
    print(f"    ✓ Generated {sequence['scenario']} sequence")
    print(f"      Timesteps: {sequence['timesteps']}")
    print(f"      Data fields: {list(sequence.keys())}")
    
    # Check data shapes
    for key, value in sequence.items():
        if isinstance(value, np.ndarray):
            print(f"      {key}: {value.shape}")
    
    return generator, sequence

def visualize_results(simulator, final_state):
    """Create simple visualizations of simulation results."""
    print("\n📈 Creating visualizations...")
    
    try:
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('FluidNetSim Demo Results', fontsize=16)
        
        # Plot density field
        if 'density' in final_state:
            im1 = axes[0, 0].imshow(final_state['density'], cmap='viridis')
            axes[0, 0].set_title('Density Field')
            axes[0, 0].set_xlabel('X')
            axes[0, 0].set_ylabel('Y')
            plt.colorbar(im1, ax=axes[0, 0])
        
        # Plot velocity magnitude
        if 'velocity_x' in final_state and 'velocity_y' in final_state:
            vel_mag = np.sqrt(final_state['velocity_x']**2 + final_state['velocity_y']**2)
            im2 = axes[0, 1].imshow(vel_mag, cmap='plasma')
            axes[0, 1].set_title('Velocity Magnitude')
            axes[0, 1].set_xlabel('X')
            axes[0, 1].set_ylabel('Y')
            plt.colorbar(im2, ax=axes[0, 1])
        
        # Plot velocity components
        if 'velocity_x' in final_state:
            im3 = axes[1, 0].imshow(final_state['velocity_x'], cmap='RdBu_r')
            axes[1, 0].set_title('X-Velocity')
            axes[1, 0].set_xlabel('X')
            axes[1, 0].set_ylabel('Y')
            plt.colorbar(im3, ax=axes[1, 0])
        
        if 'velocity_y' in final_state:
            im4 = axes[1, 1].imshow(final_state['velocity_y'], cmap='RdBu_r')
            axes[1, 1].set_title('Y-Velocity')
            axes[1, 1].set_xlabel('X')
            axes[1, 1].set_ylabel('Y')
            plt.colorbar(im4, ax=axes[1, 1])
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig('demo_results.png', dpi=150, bbox_inches='tight')
        print("    ✓ Saved visualization to 'demo_results.png'")
        
        # Show plot
        plt.show()
        
    except Exception as e:
        print(f"    ⚠ Visualization failed: {e}")

def main():
    """Main demonstration function."""
    print("🚀 FluidNetSim Basic Demonstration")
    print("=" * 50)
    
    try:
        # Run fluid simulation demo
        simulator, final_state = demo_fluid_simulation()
        
        # Test neural network models
        convlstm_model, pinn_model = demo_neural_networks()
        
        # Test training data generation
        generator, sequence = demo_training_data_generation()
        
        # Create visualizations
        visualize_results(simulator, final_state)
        
        print("\n🎉 Demo completed successfully!")
        print("\nKey Features Demonstrated:")
        print("  ✓ Advanced fluid dynamics simulation (LBM)")
        print("  ✓ Turbulence modeling (LES)")
        print("  ✓ ConvLSTM-UNet hybrid architecture")
        print("  ✓ Physics-Informed Neural Networks")
        print("  ✓ Training data generation")
        print("  ✓ Performance optimization")
        
        print("\nNext Steps:")
        print("  - Run 'python -m pytest tests/' to run all tests")
        print("  - Explore examples/ directory for more advanced usage")
        print("  - Check README.md for detailed documentation")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
