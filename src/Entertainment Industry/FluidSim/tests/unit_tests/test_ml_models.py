"""
Unit tests for FluidNetSim ML models.

Tests neural network architectures and components.
"""

import pytest
import numpy as np
import torch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from fluidnetsim.ml.models.convlstm_unet import (
    ConvLSTMUNet, ConvLSTMCell, AttentionModule, DoubleConv, Down, Up
)
from fluidnetsim.ml.models.physics_informed import PhysicsInformedNet
from fluidnetsim.ml.models.attention_mechanisms import (
    MultiHeadAttention, SpatialAttention, TemporalAttention, CrossAttention, TransformerBlock
)

class TestConvLSTMCell:
    """Test ConvLSTM cell."""
    
    def test_initialization(self):
        """Test cell initialization."""
        cell = ConvLSTMCell(input_channels=3, hidden_channels=64)
        
        assert cell.input_channels == 3
        assert cell.hidden_channels == 64
        assert cell.kernel_size == 3
        assert cell.padding == 1
        
        # Check convolution layer
        assert cell.conv.in_channels == 3 + 64  # input + hidden
        assert cell.conv.out_channels == 4 * 64  # 4 gates
    
    def test_forward_pass(self):
        """Test forward pass."""
        cell = ConvLSTMCell(input_channels=3, hidden_channels=64)
        
        batch_size, height, width = 2, 16, 16
        input_tensor = torch.randn(batch_size, 3, height, width)
        h_cur = torch.randn(batch_size, 64, height, width)
        c_cur = torch.randn(batch_size, 64, height, width)
        
        h_next, c_next = cell(input_tensor, (h_cur, c_cur))
        
        assert h_next.shape == (batch_size, 64, height, width)
        assert c_next.shape == (batch_size, 64, height, width)
        assert h_next.dtype == input_tensor.dtype
        assert c_next.dtype == input_tensor.dtype
    
    def test_hidden_state_initialization(self):
        """Test hidden state initialization."""
        cell = ConvLSTMCell(input_channels=3, hidden_channels=64)
        
        batch_size, height, width = 2, 16, 16
        device = torch.device('cpu')
        
        h, c = cell.init_hidden(batch_size, height, width, device)
        
        assert h.shape == (batch_size, 64, height, width)
        assert c.shape == (batch_size, 64, height, width)
        assert torch.all(h == 0)
        assert torch.all(c == 0)

class TestAttentionModule:
    """Test attention module."""
    
    def test_initialization(self):
        """Test attention module initialization."""
        attention = AttentionModule(channels=64)
        
        assert attention.channels == 64
        assert attention.attention_type == "spatial"
    
    def test_forward_pass(self):
        """Test forward pass."""
        attention = AttentionModule(channels=64)
        
        batch_size, channels, height, width = 2, 64, 16, 16
        x = torch.randn(batch_size, channels, height, width)
        
        output = attention(x)
        
        assert output.shape == x.shape
        assert output.dtype == x.dtype

class TestDoubleConv:
    """Test double convolution block."""
    
    def test_initialization(self):
        """Test double conv initialization."""
        conv = DoubleConv(in_channels=3, out_channels=64)
        
        assert conv.double_conv is not None
        assert conv.attention is not None
    
    def test_forward_pass(self):
        """Test forward pass."""
        conv = DoubleConv(in_channels=3, out_channels=64)
        
        batch_size, channels, height, width = 2, 3, 16, 16
        x = torch.randn(batch_size, channels, height, width)
        
        output = conv(x)
        
        assert output.shape == (batch_size, 64, height, width)
        assert output.dtype == x.dtype

class TestConvLSTMUNet:
    """Test ConvLSTM-UNet model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = ConvLSTMUNet(
            input_channels=3,
            hidden_channels=64,
            num_layers=4,
            output_channels=3
        )
        
        assert model.input_channels == 3
        assert model.hidden_channels == 64
        assert model.num_layers == 4
        assert model.output_channels == 3
        assert model.use_attention == True
        assert model.bilinear == True
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = ConvLSTMUNet(
            input_channels=3,
            hidden_channels=32,
            num_layers=3,
            output_channels=3
        )
        
        batch_size, channels, height, width = 2, 3, 32, 32
        x = torch.randn(batch_size, channels, height, width)
        
        output, hidden_states = model(x)
        
        assert output.shape == (batch_size, 3, height, width)
        assert isinstance(hidden_states, tuple)
        assert len(hidden_states) == 3  # num_layers
    
    def test_sequence_prediction(self):
        """Test sequence prediction."""
        model = ConvLSTMUNet(
            input_channels=3,
            hidden_channels=32,
            num_layers=3,
            output_channels=3
        )
        
        batch_size, channels, height, width = 2, 3, 32, 32
        initial_frames = torch.randn(batch_size, channels, height, width)
        num_steps = 5
        
        predictions = model.predict_sequence(initial_frames, num_steps)
        
        assert predictions.shape == (batch_size, num_steps, channels, height, width)
        assert predictions.dtype == initial_frames.dtype
    
    def test_model_info(self):
        """Test model information retrieval."""
        model = ConvLSTMUNet(
            input_channels=3,
            hidden_channels=32,
            num_layers=3,
            output_channels=3
        )
        
        info = model.get_model_info()
        
        assert "model_type" in info
        assert "input_channels" in info
        assert "hidden_channels" in info
        assert "num_layers" in info
        assert "output_channels" in info
        assert "total_parameters" in info
        assert "trainable_parameters" in info
        assert info["model_type"] == "ConvLSTM-UNet"
        assert info["input_channels"] == 3
        assert info["hidden_channels"] == 32

class TestPhysicsInformedNet:
    """Test Physics-Informed Neural Network."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = PhysicsInformedNet(
            input_dim=4,
            hidden_dim=128,
            num_layers=6,
            output_dim=3
        )
        
        assert model.input_dim == 4
        assert model.hidden_dim == 128
        assert model.num_layers == 6
        assert model.output_dim == 3
        assert model.activation == "tanh"
        assert model.rho == 1.0
        assert model.mu == 0.01
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = PhysicsInformedNet(
            input_dim=4,
            hidden_dim=64,
            num_layers=4,
            output_dim=3
        )
        
        batch_size, input_dim = 2, 4
        x = torch.randn(batch_size, input_dim)
        
        output = model(x)
        
        assert output.shape == (batch_size, 3)
        assert output.dtype == x.dtype
    
    def test_flow_field_prediction(self):
        """Test flow field prediction."""
        model = PhysicsInformedNet(
            input_dim=4,
            hidden_dim=64,
            num_layers=4,
            output_dim=3
        )
        
        batch_size, height, width = 2, 16, 16
        x_coords = torch.randn(batch_size, height, width)
        y_coords = torch.randn(batch_size, height, width)
        t_coords = torch.randn(batch_size, height, width)
        
        flow_fields = model.predict_flow_field(x_coords, y_coords, t_coords)
        
        assert "velocity_x" in flow_fields
        assert "velocity_y" in flow_fields
        assert "pressure" in flow_fields
        assert "velocity_magnitude" in flow_fields
        
        for field in flow_fields.values():
            assert field.shape == (batch_size, height, width)
    
    def test_physics_loss_computation(self):
        """Test physics loss computation."""
        model = PhysicsInformedNet(
            input_dim=4,
            hidden_dim=64,
            num_layers=4,
            output_dim=3
        )
        
        batch_size, height, width = 2, 16, 16
        x_coords = torch.randn(batch_size, height, width, requires_grad=True)
        y_coords = torch.randn(batch_size, height, width, requires_grad=True)
        t_coords = torch.randn(batch_size, height, width, requires_grad=True)
        
        losses = model.compute_physics_loss(x_coords, y_coords, t_coords)
        
        assert "continuity" in losses
        assert "momentum_x" in losses
        assert "momentum_y" in losses
        assert "total_physics" in losses
        
        for loss in losses.values():
            assert isinstance(loss, torch.Tensor)
            assert loss.requires_grad
    
    def test_total_loss_computation(self):
        """Test total loss computation."""
        model = PhysicsInformedNet(
            input_dim=4,
            hidden_dim=64,
            num_layers=4,
            output_dim=3
        )
        
        batch_size, height, width = 2, 16, 16
        x_coords = torch.randn(batch_size, height, width, requires_grad=True)
        y_coords = torch.randn(batch_size, height, width, requires_grad=True)
        t_coords = torch.randn(batch_size, height, width, requires_grad=True)
        
        losses = model.total_loss(x_coords, y_coords, t_coords)
        
        assert "total" in losses
        assert isinstance(losses["total"], torch.Tensor)
        assert losses["total"].requires_grad
    
    def test_physics_validation(self):
        """Test physics validation."""
        model = PhysicsInformedNet(
            input_dim=4,
            hidden_dim=64,
            num_layers=4,
            output_dim=3
        )
        
        batch_size, height, width = 2, 16, 16
        test_points = torch.randn(batch_size, height, width, 4)
        
        validation_metrics = model.validate_physics(test_points)
        
        assert "continuity" in validation_metrics
        assert "momentum_x" in validation_metrics
        assert "momentum_y" in validation_metrics
        assert "total_physics" in validation_metrics
        
        for metric in validation_metrics.values():
            assert isinstance(metric, float)
            assert metric >= 0
    
    def test_model_info(self):
        """Test model information retrieval."""
        model = PhysicsInformedNet(
            input_dim=4,
            hidden_dim=64,
            num_layers=4,
            output_dim=3
        )
        
        info = model.get_model_info()
        
        assert "model_type" in info
        assert "input_dim" in info
        assert "hidden_dim" in info
        assert "num_layers" in info
        assert "output_dim" in info
        assert "total_parameters" in info
        assert "trainable_parameters" in info
        assert "activation" in info
        assert "density" in info
        assert "viscosity" in info
        assert info["model_type"] == "PhysicsInformedNet"

class TestAttentionMechanisms:
    """Test attention mechanism classes."""
    
    def test_multihead_attention(self):
        """Test multi-head attention."""
        attention = MultiHeadAttention(embed_dim=64, num_heads=8)
        
        batch_size, seq_len, embed_dim = 2, 10, 64
        query = torch.randn(batch_size, seq_len, embed_dim)
        
        output, attn_weights = attention(query)
        
        assert output.shape == (batch_size, seq_len, embed_dim)
        assert attn_weights.shape == (batch_size, 8, seq_len, seq_len)
    
    def test_spatial_attention(self):
        """Test spatial attention."""
        attention = SpatialAttention(channels=64)
        
        batch_size, channels, height, width = 2, 64, 16, 16
        x = torch.randn(batch_size, channels, height, width)
        
        output = attention(x)
        
        assert output.shape == x.shape
        assert output.dtype == x.dtype
    
    def test_temporal_attention(self):
        """Test temporal attention."""
        attention = TemporalAttention(hidden_dim=64, num_heads=8)
        
        batch_size, seq_len, hidden_dim = 2, 10, 64
        x = torch.randn(batch_size, seq_len, hidden_dim)
        
        output, attn_weights = attention(x)
        
        assert output.shape == (batch_size, seq_len, hidden_dim)
        assert attn_weights.shape == (batch_size, 8, seq_len, seq_len)
    
    def test_cross_attention(self):
        """Test cross attention."""
        attention = CrossAttention(
            query_dim=64, key_dim=64, value_dim=64, embed_dim=64, num_heads=8
        )
        
        batch_size, seq_len, embed_dim = 2, 10, 64
        query = torch.randn(batch_size, seq_len, embed_dim)
        key = torch.randn(batch_size, seq_len, embed_dim)
        value = torch.randn(batch_size, seq_len, embed_dim)
        
        output, attn_weights = attention(query, key, value)
        
        assert output.shape == (batch_size, seq_len, embed_dim)
        assert attn_weights.shape == (batch_size, 8, seq_len, seq_len)
    
    def test_transformer_block(self):
        """Test transformer block."""
        transformer = TransformerBlock(embed_dim=64, num_heads=8, ff_dim=256)
        
        batch_size, seq_len, embed_dim = 2, 10, 64
        x = torch.randn(batch_size, seq_len, embed_dim)
        
        output = transformer(x)
        
        assert output.shape == (batch_size, seq_len, embed_dim)
        assert output.dtype == x.dtype

class TestModelIntegration:
    """Test model integration and compatibility."""
    
    def test_convlstm_unet_with_attention(self):
        """Test ConvLSTM-UNet with different attention types."""
        attention_types = ["spatial", "transformer"]
        
        for attention_type in attention_types:
            model = ConvLSTMUNet(
                input_channels=3,
                hidden_channels=32,
                num_layers=3,
                output_channels=3,
                attention_mechanism=attention_type
            )
            
            batch_size, channels, height, width = 2, 3, 32, 32
            x = torch.randn(batch_size, channels, height, width)
            
            output, hidden_states = model(x)
            
            assert output.shape == (batch_size, 3, height, width)
            assert isinstance(hidden_states, tuple)
    
    def test_physics_informed_with_different_activations(self):
        """Test PINN with different activation functions."""
        activations = ["tanh", "sin", "relu", "swish"]
        
        for activation in activations:
            model = PhysicsInformedNet(
                input_dim=4,
                hidden_dim=64,
                num_layers=4,
                output_dim=3,
                activation=activation
            )
            
            batch_size, input_dim = 2, 4
            x = torch.randn(batch_size, input_dim)
            
            output = model(x)
            
            assert output.shape == (batch_size, 3)
            assert output.dtype == x.dtype
    
    def test_model_serialization(self):
        """Test model saving and loading."""
        # Test ConvLSTM-UNet
        convlstm_model = ConvLSTMUNet(
            input_channels=3,
            hidden_channels=32,
            num_layers=3,
            output_channels=3
        )
        
        # Save model
        torch.save(convlstm_model.state_dict(), "test_convlstm.pth")
        
        # Load model
        new_model = ConvLSTMUNet(
            input_channels=3,
            hidden_channels=32,
            num_layers=3,
            output_channels=3
        )
        new_model.load_state_dict(torch.load("test_convlstm.pth"))
        
        # Test forward pass
        x = torch.randn(2, 3, 32, 32)
        output1, _ = convlstm_model(x)
        output2, _ = new_model(x)
        
        assert torch.allclose(output1, output2)
        
        # Clean up
        os.remove("test_convlstm.pth")
        
        # Test PINN
        pinn_model = PhysicsInformedNet(
            input_dim=4,
            hidden_dim=64,
            num_layers=4,
            output_dim=3
        )
        
        # Save model
        torch.save(pinn_model.state_dict(), "test_pinn.pth")
        
        # Load model
        new_pinn = PhysicsInformedNet(
            input_dim=4,
            hidden_dim=64,
            num_layers=4,
            output_dim=3
        )
        new_pinn.load_state_dict(torch.load("test_pinn.pth"))
        
        # Test forward pass
        x = torch.randn(2, 4)
        output1 = pinn_model(x)
        output2 = new_pinn(x)
        
        assert torch.allclose(output1, output2)
        
        # Clean up
        os.remove("test_pinn.pth")

if __name__ == "__main__":
    pytest.main([__file__])
