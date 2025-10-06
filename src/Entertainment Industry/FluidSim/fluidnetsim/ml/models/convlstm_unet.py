"""
ConvLSTM-UNet Hybrid Model for FluidNetSim.

Advanced neural network architecture combining ConvLSTM and U-Net for spatiotemporal modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ConvLSTMCell(nn.Module):
    """Convolutional LSTM cell for spatiotemporal modeling."""
    
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        bias: bool = True
    ):
        """
        Initialize ConvLSTM cell.
        
        Args:
            input_channels: Number of input channels
            hidden_channels: Number of hidden channels
            kernel_size: Size of convolutional kernel
            bias: Whether to use bias
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        # Gates: input, forget, cell, output
        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias
        )
    
    def forward(
        self,
        input_tensor: torch.Tensor,
        cur_state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of ConvLSTM cell.
        
        Args:
            input_tensor: Input tensor of shape (batch, channels, height, width)
            cur_state: Current (hidden, cell) state
            
        Returns:
            New (hidden, cell) state
        """
        h_cur, c_cur = cur_state
        
        # Concatenate input and hidden state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        
        # Apply convolution to get gates
        gates = self.conv(combined)
        
        # Split gates
        gates = gates.chunk(4, dim=1)
        i, f, g, o = gates
        
        # Apply activation functions
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        
        # Update cell and hidden state
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size: int, height: int, width: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state."""
        return (
            torch.zeros(batch_size, self.hidden_channels, height, width, device=device),
            torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        )

class AttentionModule(nn.Module):
    """Attention mechanism for long-range dependencies."""
    
    def __init__(self, channels: int, reduction: int = 8):
        """
        Initialize attention module.
        
        Args:
            channels: Number of channels
            reduction: Reduction factor for attention
        """
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention mechanism."""
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class DoubleConv(nn.Module):
    """Double convolution block with batch normalization and attention."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        use_attention: bool = True
    ):
        """
        Initialize double convolution block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            mid_channels: Number of intermediate channels
            use_attention: Whether to use attention mechanism
        """
        super().__init__()
        
        if mid_channels is None:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.attention = AttentionModule(out_channels) if use_attention else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.double_conv(x)
        x = self.attention(x)
        return x

class Down(nn.Module):
    """Downsampling block with max pooling."""
    
    def __init__(self, in_channels: int, out_channels: int, use_attention: bool = True):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, use_attention=use_attention)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upsampling block with skip connections."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = True,
        use_attention: bool = True
    ):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, use_attention=use_attention)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, use_attention=use_attention)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connection."""
        x1 = self.up(x1)
        
        # Handle different input sizes
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class ConvLSTMUNet(nn.Module):
    """
    ConvLSTM-UNet hybrid model for spatiotemporal fluid dynamics prediction.
    
    Features:
    - ConvLSTM layers for temporal modeling
    - U-Net architecture for spatial modeling
    - Attention mechanisms for long-range dependencies
    - Skip connections for gradient flow
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        hidden_channels: int = 64,
        num_layers: int = 4,
        output_channels: int = 3,
        attention_mechanism: str = "transformer",
        use_attention: bool = True,
        bilinear: bool = True,
        **kwargs
    ):
        """
        Initialize ConvLSTM-UNet model.
        
        Args:
            input_channels: Number of input channels (temporal frames)
            hidden_channels: Number of hidden channels
            num_layers: Number of U-Net layers
            output_channels: Number of output channels
            attention_mechanism: Type of attention mechanism
            use_attention: Whether to use attention
            bilinear: Whether to use bilinear upsampling
            **kwargs: Additional parameters
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.output_channels = output_channels
        self.attention_mechanism = attention_mechanism
        self.use_attention = use_attention
        self.bilinear = bilinear
        
        # ConvLSTM layers for temporal modeling
        self.convlstm_layers = nn.ModuleList([
            ConvLSTMCell(
                input_channels if i == 0 else hidden_channels,
                hidden_channels
            ) for i in range(num_layers)
        ])
        
        # U-Net encoder
        self.inc = DoubleConv(input_channels, hidden_channels, use_attention=use_attention)
        self.down_layers = nn.ModuleList([
            Down(hidden_channels * (2**i), hidden_channels * (2**(i+1)), use_attention=use_attention)
            for i in range(num_layers - 1)
        ])
        
        # U-Net decoder
        self.up_layers = nn.ModuleList([
            Up(
                hidden_channels * (2**(num_layers - i)),
                hidden_channels * (2**(num_layers - i - 1)),
                bilinear=bilinear,
                use_attention=use_attention
            ) for i in range(num_layers - 1)
        ])
        
        # Output layer
        self.outc = nn.Conv2d(hidden_channels, output_channels, kernel_size=1)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Initialized ConvLSTM-UNet: {input_channels}->{hidden_channels}->{output_channels}")
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of ConvLSTM-UNet.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            hidden_states: Previous hidden states (h, c)
            
        Returns:
            Output tensor and new hidden states
        """
        batch_size, channels, height, width = x.shape
        
        # Initialize hidden states if not provided
        if hidden_states is None:
            hidden_states = [
                layer.init_hidden(batch_size, height, width, x.device)
                for layer in self.convlstm_layers
            ]
        
        # Apply ConvLSTM layers
        lstm_outputs = []
        new_hidden_states = []
        
        for i, (layer, (h, c)) in enumerate(zip(self.convlstm_layers, hidden_states)):
            h_new, c_new = layer(x, (h, c))
            lstm_outputs.append(h_new)
            new_hidden_states.append((h_new, c_new))
            
            # Use output of current layer as input to next layer
            if i < len(self.convlstm_layers) - 1:
                x = h_new
        
        # U-Net forward pass using the last LSTM output
        x = lstm_outputs[-1]
        
        # Encoder path
        encoder_outputs = [self.inc(x)]
        for down_layer in self.down_layers:
            encoder_outputs.append(down_layer(encoder_outputs[-1]))
        
        # Decoder path with skip connections
        x = encoder_outputs[-1]
        for i, up_layer in enumerate(self.up_layers):
            x = up_layer(x, encoder_outputs[-(i+2)])
        
        # Output
        output = self.outc(x)
        
        return output, tuple(new_hidden_states)
    
    def predict_sequence(
        self,
        initial_frames: torch.Tensor,
        num_steps: int,
        hidden_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Predict a sequence of future frames.
        
        Args:
            initial_frames: Initial input frames (batch, channels, height, width)
            num_steps: Number of future steps to predict
            hidden_states: Initial hidden states
            
        Returns:
            Predicted sequence (batch, num_steps, channels, height, width)
        """
        batch_size = initial_frames.shape[0]
        predictions = []
        
        # Use initial frames to warm up the model
        x = initial_frames
        h = hidden_states
        
        for step in range(num_steps):
            # Forward pass
            pred, h = self.forward(x, h)
            predictions.append(pred)
            
            # Use prediction as input for next step
            x = pred
        
        return torch.stack(predictions, dim=1)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_type": "ConvLSTM-UNet",
            "input_channels": self.input_channels,
            "hidden_channels": self.hidden_channels,
            "num_layers": self.num_layers,
            "output_channels": self.output_channels,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "attention_mechanism": self.attention_mechanism,
            "use_attention": self.use_attention,
            "bilinear": self.bilinear
        }
    
    def __repr__(self) -> str:
        return (f"ConvLSTMUNet(input_channels={self.input_channels}, "
                f"hidden_channels={self.hidden_channels}, "
                f"num_layers={self.num_layers}, "
                f"output_channels={self.output_channels})")
