"""
Graph Neural Networks Implementation
This module implements various GNN architectures for graph-structured data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List
import math


class GraphConvolution(nn.Module):
    """
    Graph Convolutional Layer (GCN).
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(GraphConvolution, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Linear transformation
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through graph convolution.
        
        Args:
            x: Node features [num_nodes, in_features]
            adj: Adjacency matrix [num_nodes, num_nodes]
            
        Returns:
            Updated node features [num_nodes, out_features]
        """
        # Linear transformation
        support = torch.mm(x, self.weight)
        
        # Graph convolution: A * X * W
        output = torch.spmm(adj, support)
        
        if self.bias is not None:
            output += self.bias
        
        return output


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT).
    """
    
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 num_heads: int = 1,
                 dropout: float = 0.6,
                 alpha: float = 0.2,
                 concat: bool = True):
        super(GraphAttentionLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        # Linear transformation for each head
        self.W = nn.Parameter(torch.FloatTensor(num_heads, in_features, out_features))
        
        # Attention mechanism
        self.attention = nn.Parameter(torch.FloatTensor(num_heads, 2 * out_features, 1))
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.attention)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through graph attention layer.
        
        Args:
            x: Node features [num_nodes, in_features]
            adj: Adjacency matrix [num_nodes, num_nodes]
            
        Returns:
            Updated node features [num_nodes, out_features * num_heads] or [num_nodes, out_features]
        """
        num_nodes = x.size(0)
        
        # Linear transformation for each head
        # [num_heads, num_nodes, out_features]
        Wh = torch.stack([torch.mm(x, self.W[i]) for i in range(self.num_heads)], dim=0)
        
        # Prepare for attention computation
        # [num_heads, num_nodes, num_nodes, 2 * out_features]
        a_input = torch.cat([
            Wh.repeat_interleave(num_nodes, dim=1).view(self.num_heads, num_nodes, num_nodes, self.out_features),
            Wh.repeat(1, num_nodes, 1)
        ], dim=-1)
        
        # Compute attention scores
        # [num_heads, num_nodes, num_nodes]
        e = F.leaky_relu(torch.matmul(a_input, self.attention).squeeze(-1), negative_slope=self.alpha)
        
        # Apply mask for non-existent edges
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        # Apply softmax
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # Apply attention to values
        # [num_heads, num_nodes, out_features]
        h_prime = torch.matmul(attention, Wh)
        
        if self.concat:
            # Concatenate heads
            return h_prime.transpose(0, 1).contiguous().view(num_nodes, self.num_heads * self.out_features)
        else:
            # Average heads
            return h_prime.mean(dim=0)


class MessagePassingLayer(nn.Module):
    """
    Message Passing Neural Network Layer.
    """
    
    def __init__(self, 
                 node_features: int, 
                 edge_features: int, 
                 hidden_dim: int,
                 output_dim: int):
        super(MessagePassingLayer, self).__init__()
        
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Message function
        self.message_net = nn.Sequential(
            nn.Linear(2 * node_features + edge_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Update function
        self.update_net = nn.Sequential(
            nn.Linear(node_features + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, 
                node_features: torch.Tensor, 
                edge_features: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through message passing layer.
        
        Args:
            node_features: Node features [num_nodes, node_features]
            edge_features: Edge features [num_edges, edge_features]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Updated node features [num_nodes, output_dim]
        """
        num_nodes = node_features.size(0)
        
        # Extract source and target nodes
        row, col = edge_index
        
        # Prepare messages
        # [num_edges, 2 * node_features + edge_features]
        message_input = torch.cat([
            node_features[row],  # Source node features
            node_features[col],  # Target node features
            edge_features        # Edge features
        ], dim=1)
        
        # Compute messages
        messages = self.message_net(message_input)  # [num_edges, hidden_dim]
        
        # Aggregate messages for each node
        # [num_nodes, hidden_dim]
        aggregated_messages = torch.zeros(num_nodes, self.hidden_dim, device=node_features.device)
        aggregated_messages.index_add_(0, col, messages)
        
        # Update node features
        # [num_nodes, node_features + hidden_dim]
        update_input = torch.cat([node_features, aggregated_messages], dim=1)
        updated_features = self.update_net(update_input)
        
        return updated_features


class GraphPooling(nn.Module):
    """
    Graph pooling layer.
    """
    
    def __init__(self, pooling_type: str = 'max'):
        super(GraphPooling, self).__init__()
        
        self.pooling_type = pooling_type
    
    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Pool node features to graph-level features.
        
        Args:
            x: Node features [num_nodes, features]
            batch: Batch assignment [num_nodes]
            
        Returns:
            Graph-level features [num_graphs, features]
        """
        num_graphs = batch.max().item() + 1
        
        if self.pooling_type == 'max':
            return torch.stack([
                x[batch == i].max(dim=0)[0] for i in range(num_graphs)
            ])
        elif self.pooling_type == 'mean':
            return torch.stack([
                x[batch == i].mean(dim=0) for i in range(num_graphs)
            ])
        elif self.pooling_type == 'sum':
            return torch.stack([
                x[batch == i].sum(dim=0) for i in range(num_graphs)
            ])
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")


class GraphConvolutionalNetwork(nn.Module):
    """
    Graph Convolutional Network (GCN).
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 output_dim: int,
                 num_layers: int = 2,
                 dropout: float = 0.5):
        super(GraphConvolutionalNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList()
        
        # Input layer
        self.conv_layers.append(GraphConvolution(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.conv_layers.append(GraphConvolution(hidden_dim, hidden_dim))
        
        # Output layer
        self.conv_layers.append(GraphConvolution(hidden_dim, output_dim))
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GCN.
        
        Args:
            x: Node features [num_nodes, input_dim]
            adj: Adjacency matrix [num_nodes, num_nodes]
            
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        # Normalize adjacency matrix
        adj = self._normalize_adj(adj)
        
        # Graph convolution layers
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, adj)
            
            if i < len(self.conv_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)
        
        return x
    
    def _normalize_adj(self, adj: torch.Tensor) -> torch.Tensor:
        """Normalize adjacency matrix."""
        # Add self-loops
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        
        # Compute degree matrix
        degree = adj.sum(dim=1)
        degree_matrix = torch.diag(torch.pow(degree, -0.5))
        
        # Normalize: D^(-1/2) * A * D^(-1/2)
        normalized_adj = torch.mm(torch.mm(degree_matrix, adj), degree_matrix)
        
        return normalized_adj


class GraphAttentionNetwork(nn.Module):
    """
    Graph Attention Network (GAT).
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 output_dim: int,
                 num_heads: int = 8,
                 num_layers: int = 2,
                 dropout: float = 0.6):
        super(GraphAttentionNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Attention layers
        self.attention_layers = nn.ModuleList()
        
        # Input layer
        self.attention_layers.append(
            GraphAttentionLayer(input_dim, hidden_dim, num_heads, dropout, concat=True)
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.attention_layers.append(
                GraphAttentionLayer(hidden_dim * num_heads, hidden_dim, num_heads, dropout, concat=True)
            )
        
        # Output layer
        self.attention_layers.append(
            GraphAttentionLayer(hidden_dim * num_heads, output_dim, 1, dropout, concat=False)
        )
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GAT.
        
        Args:
            x: Node features [num_nodes, input_dim]
            adj: Adjacency matrix [num_nodes, num_nodes]
            
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        # Attention layers
        for i, attention in enumerate(self.attention_layers):
            x = attention(x, adj)
            
            if i < len(self.attention_layers) - 1:
                x = F.elu(x)
                x = F.dropout(x, self.dropout, training=self.training)
        
        return x


class MessagePassingNeuralNetwork(nn.Module):
    """
    Message Passing Neural Network (MPNN).
    """
    
    def __init__(self, 
                 node_features: int, 
                 edge_features: int,
                 hidden_dim: int, 
                 output_dim: int,
                 num_layers: int = 3):
        super(MessagePassingNeuralNetwork, self).__init__()
        
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Message passing layers
        self.mp_layers = nn.ModuleList()
        
        # Input layer
        self.mp_layers.append(
            MessagePassingLayer(node_features, edge_features, hidden_dim, hidden_dim)
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.mp_layers.append(
                MessagePassingLayer(hidden_dim, edge_features, hidden_dim, hidden_dim)
            )
        
        # Output layer
        self.mp_layers.append(
            MessagePassingLayer(hidden_dim, edge_features, hidden_dim, output_dim)
        )
        
    def forward(self, 
                node_features: torch.Tensor, 
                edge_features: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MPNN.
        
        Args:
            node_features: Node features [num_nodes, node_features]
            edge_features: Edge features [num_edges, edge_features]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        x = node_features
        
        # Message passing layers
        for mp_layer in self.mp_layers:
            x = mp_layer(x, edge_features, edge_index)
            x = F.relu(x)
        
        return x


class GraphClassifier(nn.Module):
    """
    Graph classification model.
    """
    
    def __init__(self, 
                 gnn_type: str,
                 input_dim: int, 
                 hidden_dim: int, 
                 output_dim: int,
                 num_layers: int = 3,
                 pooling_type: str = 'max'):
        super(GraphClassifier, self).__init__()
        
        self.gnn_type = gnn_type
        self.pooling_type = pooling_type
        
        # GNN backbone
        if gnn_type == 'gcn':
            self.gnn = GraphConvolutionalNetwork(input_dim, hidden_dim, hidden_dim, num_layers)
        elif gnn_type == 'gat':
            self.gnn = GraphAttentionNetwork(input_dim, hidden_dim, hidden_dim, num_layers=num_layers)
        elif gnn_type == 'mpnn':
            self.gnn = MessagePassingNeuralNetwork(input_dim, 0, hidden_dim, hidden_dim, num_layers)
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")
        
        # Pooling layer
        self.pooling = GraphPooling(pooling_type)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, 
                node_features: torch.Tensor, 
                edge_index: torch.Tensor,
                batch: torch.Tensor,
                adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for graph classification.
        
        Args:
            node_features: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment [num_nodes]
            adj: Adjacency matrix (for GCN/GAT) [num_nodes, num_nodes]
            
        Returns:
            Graph-level predictions [num_graphs, output_dim]
        """
        # GNN forward pass
        if self.gnn_type in ['gcn', 'gat']:
            if adj is None:
                # Create adjacency matrix from edge index
                num_nodes = node_features.size(0)
                adj = torch.zeros(num_nodes, num_nodes, device=node_features.device)
                adj[edge_index[0], edge_index[1]] = 1
            node_embeddings = self.gnn(node_features, adj)
        else:
            # MPNN
            edge_features = torch.zeros(edge_index.size(1), 0, device=node_features.device)
            node_embeddings = self.gnn(node_features, edge_features, edge_index)
        
        # Graph pooling
        graph_embeddings = self.pooling(node_embeddings, batch)
        
        # Classification
        output = self.classifier(graph_embeddings)
        
        return output


if __name__ == "__main__":
    # Example usage
    num_nodes = 10
    input_dim = 16
    hidden_dim = 64
    output_dim = 7
    
    # Create dummy graph data
    node_features = torch.randn(num_nodes, input_dim)
    adj = torch.randint(0, 2, (num_nodes, num_nodes)).float()
    edge_index = torch.nonzero(adj).t()
    
    # Test GCN
    gcn = GraphConvolutionalNetwork(input_dim, hidden_dim, output_dim)
    gcn_output = gcn(node_features, adj)
    print(f"GCN output shape: {gcn_output.shape}")
    
    # Test GAT
    gat = GraphAttentionNetwork(input_dim, hidden_dim, output_dim)
    gat_output = gat(node_features, adj)
    print(f"GAT output shape: {gat_output.shape}")
    
    # Test MPNN
    mpnn = MessagePassingNeuralNetwork(input_dim, 0, hidden_dim, output_dim)
    edge_features = torch.zeros(edge_index.size(1), 0)
    mpnn_output = mpnn(node_features, edge_features, edge_index)
    print(f"MPNN output shape: {mpnn_output.shape}")
    
    # Test Graph Classifier
    batch = torch.zeros(num_nodes, dtype=torch.long)  # Single graph
    classifier = GraphClassifier('gcn', input_dim, hidden_dim, output_dim)
    class_output = classifier(node_features, edge_index, batch, adj)
    print(f"Graph classifier output shape: {class_output.shape}")
