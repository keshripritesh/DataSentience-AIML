"""
Neural ODEs Implementation
This module implements Neural ODEs as continuous dynamical systems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List, Callable
import math


class ODEFunc(nn.Module):
    """
    Neural network that defines the dynamics of the ODE.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(ODEFunc, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Neural network for dynamics
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize weights
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
    
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the derivative dx/dt.
        
        Args:
            t: Time (scalar or tensor)
            x: State [batch_size, input_dim]
            
        Returns:
            Derivative dx/dt [batch_size, output_dim]
        """
        # Concatenate time and state
        if t.dim() == 0:
            t = t.expand(x.size(0), 1)
        elif t.dim() == 1:
            t = t.unsqueeze(1)
        
        # Concatenate time and state
        inputs = torch.cat([t, x], dim=1)
        
        return self.net(inputs)


class ODESolver:
    """
    Numerical ODE solver using Runge-Kutta methods.
    """
    
    def __init__(self, method: str = 'rk4'):
        self.method = method
    
    def solve(self, 
              func: Callable, 
              x0: torch.Tensor, 
              t_span: torch.Tensor) -> torch.Tensor:
        """
        Solve ODE using specified method.
        
        Args:
            func: Function that computes dx/dt
            x0: Initial state [batch_size, state_dim]
            t_span: Time points [num_times]
            
        Returns:
            Solution at all time points [num_times, batch_size, state_dim]
        """
        if self.method == 'rk4':
            return self._rk4_solve(func, x0, t_span)
        elif self.method == 'euler':
            return self._euler_solve(func, x0, t_span)
        else:
            raise ValueError(f"Unknown solver method: {self.method}")
    
    def _rk4_solve(self, 
                   func: Callable, 
                   x0: torch.Tensor, 
                   t_span: torch.Tensor) -> torch.Tensor:
        """Solve using 4th order Runge-Kutta method."""
        batch_size = x0.size(0)
        state_dim = x0.size(1)
        num_times = len(t_span)
        
        # Initialize solution
        solution = torch.zeros(num_times, batch_size, state_dim, device=x0.device)
        solution[0] = x0
        
        # Solve step by step
        for i in range(num_times - 1):
            t = t_span[i]
            dt = t_span[i + 1] - t_span[i]
            x = solution[i]
            
            # RK4 steps
            k1 = func(t, x)
            k2 = func(t + dt/2, x + dt/2 * k1)
            k3 = func(t + dt/2, x + dt/2 * k2)
            k4 = func(t + dt, x + dt * k3)
            
            # Update solution
            solution[i + 1] = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        return solution
    
    def _euler_solve(self, 
                     func: Callable, 
                     x0: torch.Tensor, 
                     t_span: torch.Tensor) -> torch.Tensor:
        """Solve using Euler method."""
        batch_size = x0.size(0)
        state_dim = x0.size(1)
        num_times = len(t_span)
        
        # Initialize solution
        solution = torch.zeros(num_times, batch_size, state_dim, device=x0.device)
        solution[0] = x0
        
        # Solve step by step
        for i in range(num_times - 1):
            t = t_span[i]
            dt = t_span[i + 1] - t_span[i]
            x = solution[i]
            
            # Euler step
            dx = func(t, x)
            solution[i + 1] = x + dt * dx
        
        return solution


class NeuralODE(nn.Module):
    """
    Neural ODE model.
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 output_dim: int,
                 solver_method: str = 'rk4'):
        super(NeuralODE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # ODE function
        self.ode_func = ODEFunc(input_dim, hidden_dim, output_dim)
        
        # ODE solver
        self.solver = ODESolver(method=solver_method)
        
        # Output projection
        self.output_projection = nn.Linear(output_dim, output_dim)
    
    def forward(self, 
                x0: torch.Tensor, 
                t_span: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Neural ODE.
        
        Args:
            x0: Initial state [batch_size, input_dim]
            t_span: Time points [num_times]
            
        Returns:
            Final state [batch_size, output_dim]
        """
        # Solve ODE
        solution = self.solver.solve(self.ode_func, x0, t_span)
        
        # Return final state
        final_state = solution[-1]
        output = self.output_projection(final_state)
        
        return output
    
    def trajectory(self, 
                  x0: torch.Tensor, 
                  t_span: torch.Tensor) -> torch.Tensor:
        """
        Get full trajectory.
        
        Args:
            x0: Initial state [batch_size, input_dim]
            t_span: Time points [num_times]
            
        Returns:
            Full trajectory [num_times, batch_size, output_dim]
        """
        solution = self.solver.solve(self.ode_func, x0, t_span)
        return self.output_projection(solution)


class AdjointNeuralODE(nn.Module):
    """
    Neural ODE with adjoint method for memory-efficient training.
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 output_dim: int,
                 solver_method: str = 'rk4'):
        super(AdjointNeuralODE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # ODE function
        self.ode_func = ODEFunc(input_dim, hidden_dim, output_dim)
        
        # ODE solver
        self.solver = ODESolver(method=solver_method)
        
        # Output projection
        self.output_projection = nn.Linear(output_dim, output_dim)
    
    def forward(self, 
                x0: torch.Tensor, 
                t_span: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with adjoint method.
        
        Args:
            x0: Initial state [batch_size, input_dim]
            t_span: Time points [num_times]
            
        Returns:
            Final state [batch_size, output_dim]
        """
        # Solve ODE forward
        solution = self.solver.solve(self.ode_func, x0, t_span)
        final_state = solution[-1]
        output = self.output_projection(final_state)
        
        # Store for backward pass
        self.solution = solution
        self.t_span = t_span
        
        return output
    
    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Backward pass using adjoint method.
        
        Args:
            grad_output: Gradient of loss w.r.t. final state
            
        Returns:
            Gradient w.r.t. initial state
        """
        # Adjoint method implementation
        # This is a simplified version - full implementation would be more complex
        
        # Compute gradient w.r.t. final state
        grad_final = self.output_projection.weight.t() @ grad_output
        
        # Backpropagate through ODE (simplified)
        grad_initial = grad_final
        
        return grad_initial


class ContinuousNormalizingFlow(nn.Module):
    """
    Continuous Normalizing Flow using Neural ODEs.
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int,
                 solver_method: str = 'rk4'):
        super(ContinuousNormalizingFlow, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Neural ODE for flow
        self.neural_ode = NeuralODE(input_dim, hidden_dim, input_dim, solver_method)
        
        # Initial distribution (standard normal)
        self.prior = torch.distributions.Normal(0, 1)
    
    def forward(self, 
                x0: torch.Tensor, 
                t_span: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform samples through the flow.
        
        Args:
            x0: Initial samples [batch_size, input_dim]
            t_span: Time points [num_times]
            
        Returns:
            Transformed samples and log probability
        """
        # Transform through ODE
        x1 = self.neural_ode(x0, t_span)
        
        # Compute log probability (simplified)
        # In practice, this would require computing the trace of the Jacobian
        log_prob = self.prior.log_prob(x0).sum(dim=1)
        
        return x1, log_prob
    
    def inverse(self, 
                x1: torch.Tensor, 
                t_span: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse transform.
        
        Args:
            x1: Final samples [batch_size, input_dim]
            t_span: Time points [num_times]
            
        Returns:
            Initial samples and log probability
        """
        # Reverse time
        t_span_reverse = torch.flip(t_span, dims=[0])
        
        # Transform back
        x0 = self.neural_ode(x1, t_span_reverse)
        
        # Compute log probability
        log_prob = self.prior.log_prob(x0).sum(dim=1)
        
        return x0, log_prob


class TimeSeriesNeuralODE(nn.Module):
    """
    Neural ODE for time series modeling.
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 output_dim: int,
                 solver_method: str = 'rk4'):
        super(TimeSeriesNeuralODE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Neural ODE
        self.neural_ode = NeuralODE(input_dim, hidden_dim, output_dim, solver_method)
        
        # Encoder for initial state
        self.encoder = nn.Linear(input_dim, input_dim)
        
        # Decoder for predictions
        self.decoder = nn.Linear(output_dim, output_dim)
    
    def forward(self, 
                x: torch.Tensor, 
                t_span: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for time series prediction.
        
        Args:
            x: Input time series [batch_size, seq_len, input_dim]
            t_span: Time points [num_times]
            
        Returns:
            Predictions [batch_size, num_times, output_dim]
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Encode initial state
        x0 = self.encoder(x[:, 0])  # Use first observation
        
        # Solve ODE
        solution = self.neural_ode.solver.solve(self.neural_ode.ode_func, x0, t_span)
        
        # Decode predictions
        predictions = self.decoder(solution)
        
        return predictions
    
    def predict_irregular(self, 
                         x: torch.Tensor, 
                         t_obs: torch.Tensor,
                         t_pred: torch.Tensor) -> torch.Tensor:
        """
        Predict at irregular time points.
        
        Args:
            x: Observations [batch_size, num_obs, input_dim]
            t_obs: Observation times [num_obs]
            t_pred: Prediction times [num_pred]
            
        Returns:
            Predictions [batch_size, num_pred, output_dim]
        """
        batch_size, num_obs, input_dim = x.shape
        
        # Encode initial state
        x0 = self.encoder(x[:, 0])
        
        # Combine observation and prediction times
        t_combined = torch.cat([t_obs, t_pred])
        t_combined, sort_idx = torch.sort(t_combined)
        
        # Solve ODE
        solution = self.neural_ode.solver.solve(self.neural_ode.ode_func, x0, t_combined)
        
        # Extract predictions
        pred_idx = sort_idx >= len(t_obs)
        predictions = solution[pred_idx]
        
        # Decode
        predictions = self.decoder(predictions)
        
        return predictions


if __name__ == "__main__":
    # Example usage
    batch_size = 4
    input_dim = 10
    hidden_dim = 64
    output_dim = 5
    num_times = 20
    
    # Create time span
    t_span = torch.linspace(0, 1, num_times)
    
    # Create Neural ODE
    neural_ode = NeuralODE(input_dim, hidden_dim, output_dim)
    
    # Create initial state
    x0 = torch.randn(batch_size, input_dim)
    
    # Forward pass
    output = neural_ode(x0, t_span)
    
    print(f"Initial state shape: {x0.shape}")
    print(f"Time span shape: {t_span.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test trajectory
    trajectory = neural_ode.trajectory(x0, t_span)
    print(f"Trajectory shape: {trajectory.shape}")
    
    # Test Continuous Normalizing Flow
    cnf = ContinuousNormalizingFlow(input_dim, hidden_dim)
    x1, log_prob = cnf(x0, t_span)
    print(f"CNF output shape: {x1.shape}")
    print(f"Log probability shape: {log_prob.shape}")
    
    # Test Time Series Neural ODE
    ts_ode = TimeSeriesNeuralODE(input_dim, hidden_dim, output_dim)
    x_series = torch.randn(batch_size, 10, input_dim)  # 10 time steps
    predictions = ts_ode(x_series, t_span)
    print(f"Time series predictions shape: {predictions.shape}")
