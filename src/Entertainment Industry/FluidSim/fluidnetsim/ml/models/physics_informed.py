"""
Physics-Informed Neural Network (PINN) for FluidNetSim.

Implements physics-constrained learning for fluid dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional, Callable
import numpy as np
import logging

logger = logging.getLogger(__name__)

class PhysicsInformedNet(nn.Module):
    """
    Physics-Informed Neural Network for fluid dynamics.
    
    Features:
    - Physics-constrained loss functions
    - Automatic differentiation for PDEs
    - Multi-objective optimization
    - Physics validation
    """
    
    def __init__(
        self,
        input_dim: int = 4,  # (x, y, t, parameters)
        hidden_dim: int = 128,
        num_layers: int = 6,
        output_dim: int = 3,  # (u, v, p)
        activation: str = "tanh",
        physics_constraints: Optional[Dict[str, Callable]] = None,
        **kwargs
    ):
        """
        Initialize Physics-Informed Neural Network.
        
        Args:
            input_dim: Input dimension (spatial + temporal + parameters)
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            output_dim: Output dimension (velocity + pressure)
            activation: Activation function
            physics_constraints: Physics constraint functions
            **kwargs: Additional parameters
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.activation = activation
        self.physics_constraints = physics_constraints or {}
        
        # Build network architecture
        self._build_network()
        
        # Physics parameters
        self.rho = 1.0  # Density
        self.mu = 0.01  # Viscosity
        
        logger.info(f"Initialized PINN: {input_dim}->{hidden_dim}->{output_dim}")
    
    def _build_network(self):
        """Build the neural network architecture."""
        layers = []
        
        # Input layer
        layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        
        # Hidden layers
        for _ in range(self.num_layers - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        
        # Output layer
        layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        
        self.network = nn.ModuleList(layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def _activation_function(self, x: torch.Tensor) -> torch.Tensor:
        """Apply activation function."""
        if self.activation == "tanh":
            return torch.tanh(x)
        elif self.activation == "sin":
            return torch.sin(x)
        elif self.activation == "relu":
            return F.relu(x)
        elif self.activation == "swish":
            return x * torch.sigmoid(x)
        else:
            return torch.tanh(x)  # Default
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.
        
        Args:
            x: Input tensor of shape (batch, input_dim)
            
        Returns:
            Output tensor of shape (batch, output_dim)
        """
        for i, layer in enumerate(self.network[:-1]):
            x = layer(x)
            x = self._activation_function(x)
        
        # Output layer (no activation)
        x = self.network[-1](x)
        return x
    
    def predict_flow_field(
        self,
        x_coords: torch.Tensor,
        y_coords: torch.Tensor,
        t_coords: torch.Tensor,
        parameters: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Predict flow field at given coordinates.
        
        Args:
            x_coords: X-coordinates
            y_coords: Y-coordinates
            t_coords: Time coordinates
            parameters: Additional parameters (viscosity, etc.)
            
        Returns:
            Dictionary containing predicted fields
        """
        # Prepare input
        if parameters is None:
            parameters = torch.zeros_like(x_coords)
        
        inputs = torch.cat([x_coords, y_coords, t_coords, parameters], dim=-1)
        
        # Forward pass
        outputs = self.forward(inputs)
        
        # Split outputs
        u = outputs[..., 0]  # x-velocity
        v = outputs[..., 1]  # y-velocity
        p = outputs[..., 2]  # pressure
        
        return {
            "velocity_x": u,
            "velocity_y": v,
            "pressure": p,
            "velocity_magnitude": torch.sqrt(u**2 + v**2)
        }
    
    def compute_physics_loss(
        self,
        x_coords: torch.Tensor,
        y_coords: torch.Tensor,
        t_coords: torch.Tensor,
        parameters: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute physics-constrained loss terms.
        
        Args:
            x_coords: X-coordinates
            y_coords: Y-coordinates
            t_coords: Time coordinates
            parameters: Additional parameters
            
        Returns:
            Dictionary containing physics loss terms
        """
        # Enable gradient computation for physics
        x_coords.requires_grad_(True)
        y_coords.requires_grad_(True)
        t_coords.requires_grad_(True)
        
        # Predict flow field
        flow_fields = self.predict_flow_field(x_coords, y_coords, t_coords, parameters)
        u = flow_fields["velocity_x"]
        v = flow_fields["velocity_y"]
        p = flow_fields["pressure"]
        
        # Compute gradients
        u_x = torch.autograd.grad(u.sum(), x_coords, create_graph=True)[0]
        u_y = torch.autograd.grad(u.sum(), y_coords, create_graph=True)[0]
        u_t = torch.autograd.grad(u.sum(), t_coords, create_graph=True)[0]
        
        v_x = torch.autograd.grad(v.sum(), x_coords, create_graph=True)[0]
        v_y = torch.autograd.grad(v.sum(), y_coords, create_graph=True)[0]
        v_t = torch.autograd.grad(v.sum(), t_coords, create_graph=True)[0]
        
        p_x = torch.autograd.grad(p.sum(), x_coords, create_graph=True)[0]
        p_y = torch.autograd.grad(p.sum(), y_coords, create_graph=True)[0]
        
        # Second derivatives for viscous terms
        u_xx = torch.autograd.grad(u_x.sum(), x_coords, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y.sum(), y_coords, create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x.sum(), x_coords, create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y.sum(), y_coords, create_graph=True)[0]
        
        # Physics loss terms
        losses = {}
        
        # Continuity equation: ∂u/∂x + ∂v/∂y = 0
        continuity_loss = torch.mean((u_x + v_y)**2)
        losses["continuity"] = continuity_loss
        
        # X-momentum equation: ∂u/∂t + u*∂u/∂x + v*∂u/∂y = -1/ρ * ∂p/∂x + ν*(∂²u/∂x² + ∂²u/∂y²)
        momentum_x_loss = torch.mean((
            u_t + u * u_x + v * u_y + 
            (1/self.rho) * p_x - 
            (self.mu/self.rho) * (u_xx + u_yy)
        )**2)
        losses["momentum_x"] = momentum_x_loss
        
        # Y-momentum equation: ∂v/∂t + u*∂v/∂x + v*∂v/∂y = -1/ρ * ∂p/∂y + ν*(∂²v/∂x² + ∂²v/∂y²)
        momentum_y_loss = torch.mean((
            v_t + u * v_x + v * v_y + 
            (1/self.rho) * p_y - 
            (self.mu/self.rho) * (v_xx + v_yy)
        )**2)
        losses["momentum_y"] = momentum_y_loss
        
        # Total physics loss
        total_physics_loss = (
            continuity_loss + 
            momentum_x_loss + 
            momentum_y_loss
        )
        losses["total_physics"] = total_physics_loss
        
        return losses
    
    def compute_boundary_loss(
        self,
        boundary_points: Dict[str, torch.Tensor],
        boundary_conditions: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute boundary condition loss.
        
        Args:
            boundary_points: Boundary point coordinates
            boundary_conditions: Boundary condition values
            
        Returns:
            Dictionary containing boundary losses
        """
        losses = {}
        
        for boundary_name, points in boundary_points.items():
            if boundary_name in boundary_conditions:
                # Extract coordinates
                x_coords = points[..., 0]
                y_coords = points[..., 1]
                t_coords = points[..., 2]
                
                # Predict at boundary
                predictions = self.predict_flow_field(x_coords, y_coords, t_coords)
                targets = boundary_conditions[boundary_name]
                
                # Compute boundary loss
                if "velocity" in boundary_name:
                    pred_vel = torch.stack([predictions["velocity_x"], predictions["velocity_y"]], dim=-1)
                    boundary_loss = F.mse_loss(pred_vel, targets)
                elif "pressure" in boundary_name:
                    boundary_loss = F.mse_loss(predictions["pressure"], targets)
                else:
                    boundary_loss = F.mse_loss(predictions["velocity_magnitude"], targets)
                
                losses[f"boundary_{boundary_name}"] = boundary_loss
        
        return losses
    
    def compute_initial_condition_loss(
        self,
        initial_points: torch.Tensor,
        initial_conditions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute initial condition loss.
        
        Args:
            initial_points: Initial condition points
            initial_conditions: Initial condition values
            
        Returns:
            Initial condition loss
        """
        x_coords = initial_points[..., 0]
        y_coords = initial_points[..., 1]
        t_coords = initial_points[..., 2]
        
        predictions = self.predict_flow_field(x_coords, y_coords, t_coords)
        pred_vel = torch.stack([predictions["velocity_x"], predictions["velocity_y"]], dim=-1)
        
        return F.mse_loss(pred_vel, initial_conditions)
    
    def total_loss(
        self,
        x_coords: torch.Tensor,
        y_coords: torch.Tensor,
        t_coords: torch.Tensor,
        parameters: Optional[torch.Tensor] = None,
        boundary_points: Optional[Dict[str, torch.Tensor]] = None,
        boundary_conditions: Optional[Dict[str, torch.Tensor]] = None,
        initial_points: Optional[torch.Tensor] = None,
        initial_conditions: Optional[torch.Tensor] = None,
        loss_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss including physics and boundary constraints.
        
        Args:
            x_coords: Interior point coordinates
            y_coords: Interior point coordinates
            t_coords: Interior point coordinates
            parameters: Additional parameters
            boundary_points: Boundary point coordinates
            boundary_conditions: Boundary condition values
            initial_points: Initial condition points
            initial_conditions: Initial condition values
            loss_weights: Weights for different loss terms
            
        Returns:
            Dictionary containing all loss terms and total loss
        """
        # Default loss weights
        if loss_weights is None:
            loss_weights = {
                "physics": 1.0,
                "boundary": 1.0,
                "initial": 1.0
            }
        
        total_loss = 0.0
        all_losses = {}
        
        # Physics loss
        if loss_weights["physics"] > 0:
            physics_losses = self.compute_physics_loss(x_coords, y_coords, t_coords, parameters)
            all_losses.update(physics_losses)
            total_loss += loss_weights["physics"] * physics_losses["total_physics"]
        
        # Boundary loss
        if boundary_points and boundary_conditions and loss_weights["boundary"] > 0:
            boundary_losses = self.compute_boundary_loss(boundary_points, boundary_conditions)
            all_losses.update(boundary_losses)
            total_loss += loss_weights["boundary"] * sum(boundary_losses.values())
        
        # Initial condition loss
        if initial_points is not None and initial_conditions is not None and loss_weights["initial"] > 0:
            initial_loss = self.compute_initial_condition_loss(initial_points, initial_conditions)
            all_losses["initial_condition"] = initial_loss
            total_loss += loss_weights["initial"] * initial_loss
        
        all_losses["total"] = total_loss
        
        return all_losses
    
    def validate_physics(self, test_points: torch.Tensor) -> Dict[str, float]:
        """
        Validate physics constraints on test points.
        
        Args:
            test_points: Test point coordinates
            
        Returns:
            Dictionary containing physics validation metrics
        """
        self.eval()
        with torch.no_grad():
            x_coords = test_points[..., 0]
            y_coords = test_points[..., 1]
            t_coords = test_points[..., 2]
            
            physics_losses = self.compute_physics_loss(x_coords, y_coords, t_coords)
            
            # Convert to numpy for evaluation
            validation_metrics = {}
            for key, loss in physics_losses.items():
                validation_metrics[key] = loss.item()
        
        self.train()
        return validation_metrics
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_type": "PhysicsInformedNet",
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "output_dim": self.output_dim,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "activation": self.activation,
            "physics_constraints": list(self.physics_constraints.keys()),
            "density": self.rho,
            "viscosity": self.mu
        }
    
    def __repr__(self) -> str:
        return (f"PhysicsInformedNet(input_dim={self.input_dim}, "
                f"hidden_dim={self.hidden_dim}, "
                f"num_layers={self.num_layers}, "
                f"output_dim={self.output_dim})")
