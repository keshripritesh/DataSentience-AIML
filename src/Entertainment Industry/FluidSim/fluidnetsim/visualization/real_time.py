"""
Real-time Visualization for FluidNetSim.

Provides interactive visualization capabilities.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)

class RealTimeVisualizer:
    """
    Real-time visualization for fluid dynamics simulation.
    
    Features:
    - Interactive plotting
    - Multiple field visualization
    - Real-time updates
    - Customizable layouts
    """
    
    def __init__(
        self,
        simulator=None,
        model=None,
        update_interval: float = 0.1,
        **kwargs
    ):
        """
        Initialize real-time visualizer.
        
        Args:
            simulator: Fluid simulator instance
            model: Neural network model
            update_interval: Update interval in seconds
            **kwargs: Additional parameters
        """
        self.simulator = simulator
        self.model = model
        self.update_interval = update_interval
        
        # Visualization state
        self.fig = None
        self.axes = None
        self.plots = {}
        self.is_running = False
        
        logger.info("Initialized real-time visualizer")
    
    def setup_plot(self, layout: str = "2x2", figsize: tuple = (12, 10)):
        """Setup the plotting layout."""
        if layout == "2x2":
            self.fig, self.axes = plt.subplots(2, 2, figsize=figsize)
            self.axes = self.axes.flatten()
        elif layout == "1x3":
            self.fig, self.axes = plt.subplots(1, 3, figsize=figsize)
        else:
            self.fig, self.axes = plt.subplots(1, 1, figsize=figsize)
            self.axes = [self.axes]
        
        plt.ion()  # Enable interactive mode
        logger.info(f"Setup plot with layout: {layout}")
    
    def plot_field(self, field: np.ndarray, title: str, ax_idx: int = 0, 
                   cmap: str = "viridis", **kwargs):
        """Plot a single field."""
        if self.axes is None:
            self.setup_plot()
        
        ax = self.axes[ax_idx]
        ax.clear()
        
        im = ax.imshow(field, cmap=cmap, **kwargs)
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Store plot reference
        self.plots[f"{title}_{ax_idx}"] = im
        
        return im
    
    def plot_velocity_field(self, velocity_x: np.ndarray, velocity_y: np.ndarray, 
                           ax_idx: int = 0, **kwargs):
        """Plot velocity field with streamlines."""
        if self.axes is None:
            self.setup_plot()
        
        ax = self.axes[ax_idx]
        ax.clear()
        
        # Plot velocity magnitude
        velocity_magnitude = np.sqrt(velocity_x**2 + velocity_y**2)
        im = ax.imshow(velocity_magnitude, cmap='plasma', **kwargs)
        
        # Add streamlines
        y, x = np.mgrid[0:velocity_x.shape[0], 0:velocity_x.shape[1]]
        ax.streamplot(x, y, velocity_x, velocity_y, color='white', alpha=0.6, density=1.5)
        
        ax.set_title('Velocity Field with Streamlines')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        self.plots[f"velocity_{ax_idx}"] = im
        return im
    
    def update_plot(self, data: Dict[str, np.ndarray]):
        """Update all plots with new data."""
        if self.fig is None:
            return
        
        # Update density field
        if 'density' in data:
            self.plot_field(data['density'], 'Density Field', 0, 'viridis')
        
        # Update velocity field
        if 'velocity_x' in data and 'velocity_y' in data:
            self.plot_velocity_field(data['velocity_x'], data['velocity_y'], 1)
        
        # Update pressure field
        if 'pressure' in data:
            self.plot_field(data['pressure'], 'Pressure Field', 2, 'RdBu_r')
        
        # Update any other fields
        if 'velocity_magnitude' in data:
            self.plot_field(data['velocity_magnitude'], 'Velocity Magnitude', 3, 'plasma')
        
        # Refresh display
        plt.tight_layout()
        plt.pause(0.001)
    
    def run_interactive(self, num_steps: int = 100, dt: float = 0.01):
        """Run interactive simulation with real-time visualization."""
        if self.simulator is None:
            logger.error("No simulator provided")
            return
        
        self.setup_plot()
        self.is_running = True
        
        try:
            for step in range(num_steps):
                if not self.is_running:
                    break
                
                # Run simulation step
                state = self.simulator.step(dt)
                
                # Update visualization
                self.update_plot(state)
                
                # Update title with current time
                if self.fig:
                    self.fig.suptitle(f'FluidNetSim - Time: {state.get("time", 0):.3f}s, Step: {step+1}')
                
                plt.pause(self.update_interval)
                
        except KeyboardInterrupt:
            logger.info("Interactive simulation interrupted by user")
        finally:
            self.is_running = False
            plt.ioff()
    
    def compare_prediction(self, true_data: Dict[str, np.ndarray], 
                          predicted_data: Dict[str, np.ndarray]):
        """Compare true vs predicted data side by side."""
        if self.fig is None:
            self.setup_plot("1x3", (18, 6))
        
        # Plot true data
        if 'velocity_magnitude' in true_data:
            self.plot_field(true_data['velocity_magnitude'], 'True Velocity Magnitude', 0, 'plasma')
        
        # Plot predicted data
        if 'velocity_magnitude' in predicted_data:
            self.plot_field(predicted_data['velocity_magnitude'], 'Predicted Velocity Magnitude', 1, 'plasma')
        
        # Plot difference
        if 'velocity_magnitude' in true_data and 'velocity_magnitude' in predicted_data:
            diff = true_data['velocity_magnitude'] - predicted_data['velocity_magnitude']
            self.plot_field(diff, 'Difference (True - Predicted)', 2, 'RdBu_r')
        
        plt.tight_layout()
        plt.show()
    
    def create_animation(self, data_sequence: list, output_path: str = "animation.gif"):
        """Create animation from data sequence."""
        try:
            import matplotlib.animation as animation
            
            if self.fig is None:
                self.setup_plot()
            
            def animate(frame):
                data = data_sequence[frame]
                self.update_plot(data)
                return [self.plots.get(key) for key in self.plots.keys()]
            
            anim = animation.FuncAnimation(
                self.fig, animate, frames=len(data_sequence),
                interval=100, blit=False
            )
            
            # Save animation
            anim.save(output_path, writer='pillow', fps=10)
            logger.info(f"Animation saved to {output_path}")
            
        except ImportError:
            logger.warning("matplotlib.animation not available, cannot create animation")
    
    def stop(self):
        """Stop the visualization."""
        self.is_running = False
        if self.fig:
            plt.close(self.fig)
    
    def __repr__(self) -> str:
        return f"RealTimeVisualizer(simulator={self.simulator is not None}, model={self.model is not None})"
