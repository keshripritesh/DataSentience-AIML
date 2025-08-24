"""
Neural Turing Machine Implementation
This module implements a complete NTM with external memory and attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List
import math


class MemoryBank(nn.Module):
    """
    External memory bank for the Neural Turing Machine.
    """
    
    def __init__(self, memory_size: int, memory_width: int):
        super(MemoryBank, self).__init__()
        
        self.memory_size = memory_size
        self.memory_width = memory_width
        
        # Initialize memory matrix
        self.memory = nn.Parameter(torch.randn(memory_size, memory_width) * 0.1)
        
    def read(self, read_weights: torch.Tensor) -> torch.Tensor:
        """
        Read from memory using attention weights.
        
        Args:
            read_weights: Attention weights [batch_size, memory_size]
            
        Returns:
            Read vector [batch_size, memory_width]
        """
        # Weighted sum of memory locations
        read_vector = torch.matmul(read_weights, self.memory)
        return read_vector
    
    def write(self, write_weights: torch.Tensor, erase_vector: torch.Tensor, add_vector: torch.Tensor):
        """
        Write to memory using attention weights.
        
        Args:
            write_weights: Attention weights [batch_size, memory_size]
            erase_vector: Erase vector [batch_size, memory_width]
            add_vector: Add vector [batch_size, memory_width]
        """
        # Expand weights for broadcasting
        write_weights_expanded = write_weights.unsqueeze(-1)  # [batch_size, memory_size, 1]
        
        # Erase operation: memory = memory * (1 - w * e)
        erase_term = write_weights_expanded * erase_vector.unsqueeze(1)  # [batch_size, memory_size, memory_width]
        self.memory.data = self.memory.data * (1 - erase_term.sum(dim=0))
        
        # Add operation: memory = memory + w * a
        add_term = write_weights_expanded * add_vector.unsqueeze(1)  # [batch_size, memory_size, memory_width]
        self.memory.data = self.memory.data + add_term.sum(dim=0)


class ReadWriteHeads(nn.Module):
    """
    Read and write heads with attention mechanisms.
    """
    
    def __init__(self, 
                 controller_output_size: int,
                 memory_size: int,
                 memory_width: int,
                 num_read_heads: int = 1,
                 num_write_heads: int = 1):
        super(ReadWriteHeads, self).__init__()
        
        self.memory_size = memory_size
        self.memory_width = memory_width
        self.num_read_heads = num_read_heads
        self.num_write_heads = num_write_heads
        
        # Key, beta, and gamma parameters for content-based addressing
        self.read_keys = nn.Linear(controller_output_size, num_read_heads * memory_width)
        self.read_betas = nn.Linear(controller_output_size, num_read_heads)
        self.read_gammas = nn.Linear(controller_output_size, num_read_heads)
        
        self.write_keys = nn.Linear(controller_output_size, num_write_heads * memory_width)
        self.write_betas = nn.Linear(controller_output_size, num_write_heads)
        self.write_gammas = nn.Linear(controller_output_size, num_write_heads)
        
        # Write vectors
        self.erase_vectors = nn.Linear(controller_output_size, num_write_heads * memory_width)
        self.add_vectors = nn.Linear(controller_output_size, num_write_heads * memory_width)
        
        # Shifting parameters for location-based addressing
        self.read_shifts = nn.Linear(controller_output_size, num_read_heads * 3)  # -1, 0, 1
        self.write_shifts = nn.Linear(controller_output_size, num_write_heads * 3)
        
        # Sharpening parameters
        self.read_sharpenings = nn.Linear(controller_output_size, num_read_heads)
        self.write_sharpenings = nn.Linear(controller_output_size, num_write_heads)
        
    def forward(self, controller_output: torch.Tensor, prev_read_weights: List[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute read and write operations.
        
        Args:
            controller_output: Output from controller [batch_size, controller_output_size]
            prev_read_weights: Previous read weights for each head
            
        Returns:
            Dictionary containing read vectors, read weights, write weights, etc.
        """
        batch_size = controller_output.size(0)
        
        # Initialize previous weights if not provided
        if prev_read_weights is None:
            prev_read_weights = [torch.ones(batch_size, self.memory_size, device=controller_output.device) / self.memory_size 
                               for _ in range(self.num_read_heads)]
        
        # Content-based addressing for read heads
        read_keys = self.read_keys(controller_output).view(batch_size, self.num_read_heads, self.memory_width)
        read_betas = F.softplus(self.read_betas(controller_output)).view(batch_size, self.num_read_heads)
        read_gammas = 1 + F.softplus(self.read_gammas(controller_output)).view(batch_size, self.num_read_heads)
        
        read_weights = []
        for i in range(self.num_read_heads):
            weights = self._content_based_addressing(
                read_keys[:, i], read_betas[:, i], read_gammas[:, i], prev_read_weights[i]
            )
            read_weights.append(weights)
        
        # Content-based addressing for write heads
        write_keys = self.write_keys(controller_output).view(batch_size, self.num_write_heads, self.memory_width)
        write_betas = F.softplus(self.write_betas(controller_output)).view(batch_size, self.num_write_heads)
        write_gammas = 1 + F.softplus(self.write_gammas(controller_output)).view(batch_size, self.num_write_heads)
        
        write_weights = []
        for i in range(self.num_write_heads):
            weights = self._content_based_addressing(
                write_keys[:, i], write_betas[:, i], write_gammas[:, i], 
                torch.ones(batch_size, self.memory_size, device=controller_output.device) / self.memory_size
            )
            write_weights.append(weights)
        
        # Write vectors
        erase_vectors = torch.sigmoid(self.erase_vectors(controller_output)).view(batch_size, self.num_write_heads, self.memory_width)
        add_vectors = torch.tanh(self.add_vectors(controller_output)).view(batch_size, self.num_write_heads, self.memory_width)
        
        return {
            'read_keys': read_keys,
            'read_betas': read_betas,
            'read_gammas': read_gammas,
            'read_weights': read_weights,
            'write_keys': write_keys,
            'write_betas': write_betas,
            'write_gammas': write_gammas,
            'write_weights': write_weights,
            'erase_vectors': erase_vectors,
            'add_vectors': add_vectors
        }
    
    def _content_based_addressing(self, 
                                 key: torch.Tensor, 
                                 beta: torch.Tensor, 
                                 gamma: torch.Tensor,
                                 prev_weights: torch.Tensor) -> torch.Tensor:
        """
        Content-based addressing mechanism.
        
        Args:
            key: Key vector [batch_size, memory_width]
            beta: Key strength [batch_size]
            gamma: Interpolation gate [batch_size]
            prev_weights: Previous attention weights [batch_size, memory_size]
            
        Returns:
            Attention weights [batch_size, memory_size]
        """
        # Compute cosine similarity between key and memory
        # Assuming memory is stored as a parameter in the parent module
        memory = self.parent.memory if hasattr(self, 'parent') else torch.randn(self.memory_size, self.memory_width)
        
        # Normalize key and memory for cosine similarity
        key_norm = F.normalize(key, p=2, dim=1)
        memory_norm = F.normalize(memory, p=2, dim=1)
        
        # Compute similarity
        similarity = torch.matmul(key_norm, memory_norm.t())  # [batch_size, memory_size]
        
        # Apply key strength
        weights = F.softmax(beta.unsqueeze(1) * similarity, dim=1)
        
        # Interpolation with previous weights
        weights = gamma.unsqueeze(1) * weights + (1 - gamma.unsqueeze(1)) * prev_weights
        
        return weights


class NTMController(nn.Module):
    """
    Controller network for the Neural Turing Machine.
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 controller_type: str = 'lstm'):
        super(NTMController, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.controller_type = controller_type
        
        if controller_type == 'lstm':
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.output_projection = nn.Linear(hidden_size, output_size)
        elif controller_type == 'feedforward':
            self.feedforward = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
            )
        else:
            raise ValueError(f"Unknown controller type: {controller_type}")
    
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through the controller.
        
        Args:
            x: Input [batch_size, seq_len, input_size] or [batch_size, input_size]
            hidden: LSTM hidden state (if using LSTM)
            
        Returns:
            Output and new hidden state
        """
        if self.controller_type == 'lstm':
            if x.dim() == 2:
                x = x.unsqueeze(1)  # Add sequence dimension
            
            lstm_out, hidden = self.lstm(x, hidden)
            output = self.output_projection(lstm_out)
            
            if x.size(1) == 1:
                output = output.squeeze(1)  # Remove sequence dimension
                
            return output, hidden
        else:
            output = self.feedforward(x)
            return output, None


class NeuralTuringMachine(nn.Module):
    """
    Complete Neural Turing Machine implementation.
    """
    
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 memory_size: int = 128,
                 memory_width: int = 20,
                 controller_hidden_size: int = 100,
                 num_read_heads: int = 1,
                 num_write_heads: int = 1,
                 controller_type: str = 'lstm'):
        super(NeuralTuringMachine, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.memory_size = memory_size
        self.memory_width = memory_width
        self.num_read_heads = num_read_heads
        self.num_write_heads = num_write_heads
        self.controller_type = controller_type
        
        # Controller
        controller_output_size = output_size + num_read_heads * memory_width + num_write_heads * (2 * memory_width + 3)
        self.controller = NTMController(
            input_size=input_size + num_read_heads * memory_width,  # Input + read vectors
            hidden_size=controller_hidden_size,
            output_size=controller_output_size,
            controller_type=controller_type
        )
        
        # Memory bank
        self.memory_bank = MemoryBank(memory_size, memory_width)
        
        # Read/write heads
        self.heads = ReadWriteHeads(
            controller_output_size=controller_output_size,
            memory_size=memory_size,
            memory_width=memory_width,
            num_read_heads=num_read_heads,
            num_write_heads=num_write_heads
        )
        
        # Connect heads to memory bank
        self.heads.parent = self.memory_bank
        
        # Output projection
        self.output_projection = nn.Linear(output_size, output_size)
        
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the NTM.
        
        Args:
            x: Input sequence [batch_size, seq_len, input_size] or single input [batch_size, input_size]
            hidden: Controller hidden state
            
        Returns:
            Dictionary containing outputs and internal states
        """
        batch_size = x.size(0)
        is_sequence = x.dim() == 3
        
        if is_sequence:
            seq_len = x.size(1)
            outputs = []
            read_vectors_list = []
            write_weights_list = []
            read_weights_list = []
            
            # Initialize read weights
            read_weights = [torch.ones(batch_size, self.memory_size, device=x.device) / self.memory_size 
                          for _ in range(self.num_read_heads)]
            
            for t in range(seq_len):
                # Read from memory
                read_vectors = []
                for i in range(self.num_read_heads):
                    read_vector = self.memory_bank.read(read_weights[i])
                    read_vectors.append(read_vector)
                
                # Concatenate input with read vectors
                read_vectors_concat = torch.cat(read_vectors, dim=1)
                controller_input = torch.cat([x[:, t], read_vectors_concat], dim=1)
                
                # Controller forward pass
                controller_output, hidden = self.controller(controller_input, hidden)
                
                # Parse controller output
                output_size = self.output_size
                read_size = self.num_read_heads * self.memory_width
                write_size = self.num_write_heads * (2 * self.memory_width + 3)
                
                output = controller_output[:, :output_size]
                read_params = controller_output[:, output_size:output_size + read_size]
                write_params = controller_output[:, output_size + read_size:]
                
                # Read/write heads
                head_outputs = self.heads(controller_output, read_weights)
                
                # Write to memory
                for i in range(self.num_write_heads):
                    self.memory_bank.write(
                        head_outputs['write_weights'][i],
                        head_outputs['erase_vectors'][:, i],
                        head_outputs['add_vectors'][:, i]
                    )
                
                # Update read weights
                read_weights = head_outputs['read_weights']
                
                # Store outputs
                outputs.append(output)
                read_vectors_list.append(read_vectors)
                write_weights_list.append(head_outputs['write_weights'])
                read_weights_list.append(read_weights)
            
            # Stack outputs
            outputs = torch.stack(outputs, dim=1)
            
        else:
            # Single timestep
            # Read from memory
            read_vectors = []
            read_weights = [torch.ones(batch_size, self.memory_size, device=x.device) / self.memory_size 
                          for _ in range(self.num_read_heads)]
            
            for i in range(self.num_read_heads):
                read_vector = self.memory_bank.read(read_weights[i])
                read_vectors.append(read_vector)
            
            # Concatenate input with read vectors
            read_vectors_concat = torch.cat(read_vectors, dim=1)
            controller_input = torch.cat([x, read_vectors_concat], dim=1)
            
            # Controller forward pass
            controller_output, hidden = self.controller(controller_input, hidden)
            
            # Parse output
            output = controller_output[:, :self.output_size]
            outputs = output
            read_vectors_list = [read_vectors]
            write_weights_list = []
            read_weights_list = []
        
        return {
            'output': outputs,
            'read_vectors': read_vectors_list,
            'write_weights': write_weights_list,
            'read_weights': read_weights_list,
            'hidden': hidden
        }


if __name__ == "__main__":
    # Example usage
    batch_size = 4
    seq_len = 10
    input_size = 8
    output_size = 4
    
    # Create NTM
    ntm = NeuralTuringMachine(
        input_size=input_size,
        output_size=output_size,
        memory_size=128,
        memory_width=20,
        controller_hidden_size=100,
        num_read_heads=1,
        num_write_heads=1,
        controller_type='lstm'
    )
    
    # Create input sequence
    x = torch.randn(batch_size, seq_len, input_size)
    
    # Forward pass
    outputs = ntm(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {outputs['output'].shape}")
    print(f"Memory shape: {ntm.memory_bank.memory.shape}")
    
    # Test single timestep
    x_single = torch.randn(batch_size, input_size)
    outputs_single = ntm(x_single)
    
    print(f"Single input shape: {x_single.shape}")
    print(f"Single output shape: {outputs_single['output'].shape}")
