# Graph Neural Networks (GNNs)

## Overview
Graph Neural Networks (GNNs) represent a fundamental shift in deep learning by operating directly on graph-structured data. Unlike traditional neural networks that process fixed-dimensional vectors, GNNs can handle irregular, relational data structures where entities (nodes) are connected through relationships (edges). This enables learning from complex networks, social graphs, molecular structures, knowledge graphs, and any data that can be represented as a graph.

## Core Concepts

### Graph Structure
A graph G = (V, E) consists of:
- **Nodes (Vertices)**: Entities with feature vectors xᵥ ∈ ℝᵈ
- **Edges**: Relationships between nodes with optional edge features eᵤᵥ ∈ ℝᵉ
- **Adjacency Matrix**: A ∈ {0,1}ⁿˣⁿ representing connections

**Key Insight:** GNNs preserve the graph structure while learning node representations that capture both local and global information.

### Message Passing Framework
The core mechanism of GNNs is message passing, where nodes exchange information with their neighbors:

```
hᵥ⁽ˡ⁺¹⁾ = UPDATE⁽ˡ⁾(hᵥ⁽ˡ⁾, AGGREGATE⁽ˡ⁾({hᵤ⁽ˡ⁾ : u ∈ N(v)}))
```

Where:
- `hᵥ⁽ˡ⁾` is the representation of node v at layer l
- `N(v)` is the neighborhood of node v
- `AGGREGATE` combines neighbor messages
- `UPDATE` updates node representation

### Graph Convolution
Graph convolution generalizes convolutional operations to irregular graph structures:

**Spectral Convolution:**
```
H⁽ˡ⁺¹⁾ = σ(D⁻¹/²AD⁻¹/²H⁽ˡ⁾W⁽ˡ⁾)
```

**Spatial Convolution:**
```
hᵥ⁽ˡ⁺¹⁾ = σ(W⁽ˡ⁾ · AGGREGATE({hᵤ⁽ˡ⁾ : u ∈ N(v)}))
```

### Attention Mechanisms
Graph attention enables nodes to selectively attend to their neighbors:

```
αᵢⱼ = softmax(LeakyReLU(aᵀ[Whᵢ || Whⱼ]))
hᵢ' = σ(∑ⱼ∈N(i) αᵢⱼWhⱼ)
```

Where `αᵢⱼ` is the attention weight between nodes i and j.

## Bizarre and Advanced Aspects

### 1. Irregular Data Processing
GNNs can handle data with irregular structure, unlike CNNs (grids) or RNNs (sequences). This makes them suitable for complex relational data.

### 2. Inductive vs Transductive Learning
- **Transductive**: Can only make predictions for nodes seen during training
- **Inductive**: Can generalize to new nodes and graphs not seen during training

### 3. Permutation Invariance
GNNs are invariant to node ordering, meaning the same graph with different node orderings produces the same output.

### 4. Multi-Scale Representations
GNNs can capture information at multiple scales:
- **Local**: Direct neighbor information
- **Global**: Information from distant nodes through message passing

### 5. Heterogeneous Graphs
Advanced GNNs can handle graphs with different types of nodes and edges, enabling modeling of complex multi-relational data.

### 6. Dynamic Graphs
GNNs can model evolving graphs where nodes and edges change over time.

## Technical Architecture

### Graph Convolutional Network (GCN)
```python
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        # Normalize adjacency matrix
        adj_normalized = self.normalize_adj(adj)
        
        # Graph convolution: H' = σ(D⁻¹/²AD⁻¹/²HW)
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj_normalized, support)
        
        if self.bias is not None:
            output += self.bias
        
        return F.relu(output)
    
    def normalize_adj(self, adj):
        # D⁻¹/²AD⁻¹/² normalization
        adj = adj + torch.eye(adj.size(0)).to(adj.device)
        deg = torch.sum(adj, dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        return deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
```

### Graph Attention Network (GAT)
```python
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, n_heads=1, dropout=0.6, alpha=0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.dropout = dropout
        self.alpha = alpha
        
        # Linear transformation for each head
        self.W = nn.Parameter(torch.FloatTensor(n_heads, in_features, out_features))
        
        # Attention mechanism
        self.attention = nn.Parameter(torch.FloatTensor(n_heads, 2 * out_features, 1))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.attention)
    
    def forward(self, x, adj):
        batch_size = x.size(0)
        
        # Linear transformation for each head
        Wh = torch.stack([torch.mm(x, self.W[i]) for i in range(self.n_heads)], dim=1)
        
        # Prepare for attention computation
        a_input = torch.cat([
            Wh.repeat_interleave(batch_size, dim=0),
            Wh.repeat(batch_size, 1, 1)
        ], dim=2).view(batch_size, batch_size, self.n_heads, 2 * self.out_features)
        
        # Compute attention scores
        e = F.leaky_relu(torch.sum(a_input * self.attention, dim=3))
        
        # Mask attention scores for non-adjacent nodes
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        # Apply softmax
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # Apply attention to values
        h_prime = torch.sum(attention.unsqueeze(-1) * Wh.unsqueeze(1), dim=2)
        
        return h_prime
```

### Message Passing Neural Network (MPNN)
```python
class MessagePassing(nn.Module):
    def __init__(self, node_features, edge_features, hidden_dim, output_dim):
        super().__init__()
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
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output function
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def message(self, h_i, h_j, e_ij):
        # Compute messages from neighbors
        message_input = torch.cat([h_i, h_j, e_ij], dim=-1)
        return self.message_net(message_input)
    
    def update(self, h_i, messages):
        # Update node representations
        update_input = torch.cat([h_i, messages], dim=-1)
        return self.update_net(update_input)
    
    def forward(self, x, edge_index, edge_attr, num_layers=3):
        h = x
        
        for layer in range(num_layers):
            # Aggregate messages from neighbors
            messages = self.aggregate_messages(h, edge_index, edge_attr)
            
            # Update node representations
            h = self.update(h, messages)
        
        # Final output
        return self.output_net(h)
    
    def aggregate_messages(self, h, edge_index, edge_attr):
        # Aggregate messages using edge_index
        row, col = edge_index
        messages = self.message(h[row], h[col], edge_attr)
        
        # Sum messages for each node
        aggregated = torch.zeros_like(h)
        aggregated.index_add_(0, row, messages)
        
        return aggregated
```

## Implementation Details

### Basic GNN Architecture
```python
class GraphNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GraphConvolution(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.conv_layers.append(GraphConvolution(hidden_dim, hidden_dim))
        
        self.conv_layers.append(GraphConvolution(hidden_dim, output_dim))
        
        self.dropout = dropout
    
    def forward(self, x, adj):
        # Apply graph convolutions
        for i, conv in enumerate(self.conv_layers[:-1]):
            x = conv(x, adj)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        
        # Final layer without activation
        x = self.conv_layers[-1](x, adj)
        
        return x
```

### Graph Pooling
```python
class GraphPooling(nn.Module):
    def __init__(self, pooling_ratio=0.5):
        super().__init__()
        self.pooling_ratio = pooling_ratio
    
    def forward(self, x, adj, batch=None):
        # Compute node scores for pooling
        scores = self.compute_scores(x)
        
        # Select top nodes
        num_nodes = x.size(0)
        num_pool = int(num_nodes * self.pooling_ratio)
        
        _, indices = torch.topk(scores, num_pool)
        
        # Pool nodes and adjacency matrix
        x_pooled = x[indices]
        adj_pooled = adj[indices][:, indices]
        
        return x_pooled, adj_pooled, indices
    
    def compute_scores(self, x):
        # Simple scoring based on node features
        return torch.sum(x, dim=1)
```

### Graph Classification
```python
class GraphClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=3):
        super().__init__()
        self.gnn = GraphNeuralNetwork(input_dim, hidden_dim, hidden_dim, num_layers)
        self.pooling = GraphPooling()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x, adj, batch=None):
        # Apply GNN
        x = self.gnn(x, adj)
        
        # Pool graph
        x_pooled, adj_pooled, _ = self.pooling(x, adj, batch)
        
        # Global pooling (mean)
        x_global = torch.mean(x_pooled, dim=0)
        
        # Classify
        return self.classifier(x_global)
```

## Advanced Variants

### 1. GraphSAGE
Inductive learning on large graphs using neighborhood sampling:

```python
class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList()
        self.layers.append(GraphSAGELayer(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.layers.append(GraphSAGELayer(hidden_dim, hidden_dim))
        
        self.layers.append(GraphSAGELayer(hidden_dim, output_dim))
    
    def forward(self, x, adj, num_samples=None):
        h = x
        
        for layer in self.layers:
            h = layer(h, adj, num_samples)
        
        return h

class GraphSAGELayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features * 2, out_features)
    
    def forward(self, x, adj, num_samples=None):
        # Sample neighbors if specified
        if num_samples is not None:
            adj = self.sample_neighbors(adj, num_samples)
        
        # Aggregate neighbor features
        neighbor_features = torch.spmm(adj, x)
        
        # Concatenate with self features
        combined = torch.cat([x, neighbor_features], dim=1)
        
        return F.relu(self.linear(combined))
```

### 2. Heterogeneous Graph Neural Network
Handles graphs with multiple node and edge types:

```python
class HeterogeneousGNN(nn.Module):
    def __init__(self, node_types, edge_types, hidden_dim):
        super().__init__()
        self.node_types = node_types
        self.edge_types = edge_types
        
        # Node type-specific encoders
        self.node_encoders = nn.ModuleDict({
            node_type: nn.Linear(input_dim, hidden_dim)
            for node_type, input_dim in node_types.items()
        })
        
        # Edge type-specific message functions
        self.message_functions = nn.ModuleDict({
            edge_type: nn.Linear(2 * hidden_dim, hidden_dim)
            for edge_type in edge_types
        })
        
        # Node type-specific update functions
        self.update_functions = nn.ModuleDict({
            node_type: nn.Linear(2 * hidden_dim, hidden_dim)
            for node_type in node_types
        })
    
    def forward(self, node_features, edge_index, edge_type, node_type):
        # Encode node features by type
        h = {}
        for n_type in self.node_types:
            mask = (node_type == n_type)
            if mask.sum() > 0:
                h[n_type] = self.node_encoders[n_type](node_features[mask])
        
        # Message passing for each edge type
        for e_type in self.edge_types:
            edge_mask = (edge_type == e_type)
            if edge_mask.sum() > 0:
                messages = self.compute_messages(h, edge_index[:, edge_mask], e_type)
                h = self.update_nodes(h, messages, node_type)
        
        return h
```

### 3. Temporal Graph Neural Network
Models evolving graphs over time:

```python
class TemporalGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.gnn = GraphNeuralNetwork(input_dim, hidden_dim, hidden_dim, num_layers)
        self.temporal_encoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.predictor = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, node_features, adj_matrices, timestamps):
        # Process each time step
        temporal_features = []
        
        for t in range(len(adj_matrices)):
            # Apply GNN at time t
            h_t = self.gnn(node_features, adj_matrices[t])
            temporal_features.append(h_t)
        
        # Encode temporal sequence
        temporal_features = torch.stack(temporal_features, dim=1)
        lstm_out, _ = self.temporal_encoder(temporal_features)
        
        # Predict next time step
        predictions = self.predictor(lstm_out)
        
        return predictions
```

### 4. Graph Neural Network with External Memory
Incorporates external memory for better representation learning:

```python
class MemoryGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, memory_size, memory_dim):
        super().__init__()
        self.gnn = GraphNeuralNetwork(input_dim, hidden_dim, hidden_dim)
        self.memory = nn.Parameter(torch.randn(memory_size, memory_dim))
        self.memory_attention = nn.MultiheadAttention(memory_dim, num_heads=4)
        self.output_layer = nn.Linear(hidden_dim + memory_dim, hidden_dim)
    
    def forward(self, x, adj):
        # Apply GNN
        h = self.gnn(x, adj)
        
        # Read from memory
        memory_read, _ = self.memory_attention(h, self.memory, self.memory)
        
        # Combine with GNN output
        combined = torch.cat([h, memory_read], dim=-1)
        output = self.output_layer(combined)
        
        return output
```

## Performance Metrics

### 1. Node Classification Metrics
- **Accuracy**: Classification accuracy for node labels
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the receiver operating characteristic curve

### 2. Graph Classification Metrics
- **Accuracy**: Classification accuracy for graph labels
- **Cross-validation**: Performance across different graph splits
- **Transfer learning**: Performance on unseen graph domains

### 3. Link Prediction Metrics
- **AUC**: Area under the curve for link prediction
- **Precision@K**: Precision at top-K predictions
- **MRR**: Mean reciprocal rank

### 4. Scalability Metrics
- **Training time**: Time per epoch
- **Memory usage**: GPU/CPU memory consumption
- **Inference time**: Time per forward pass

## Applications

### 1. Social Network Analysis
- **Community detection**: Identifying groups of related users
- **Influence prediction**: Predicting information spread
- **Recommendation systems**: User-item recommendations

### 2. Molecular Biology
- **Drug discovery**: Predicting molecular properties
- **Protein interaction**: Modeling protein-protein interactions
- **Chemical reaction**: Predicting reaction outcomes

### 3. Computer Vision
- **Scene understanding**: Modeling object relationships
- **Point cloud processing**: 3D point cloud analysis
- **Image segmentation**: Hierarchical segmentation

### 4. Natural Language Processing
- **Knowledge graphs**: Entity and relation extraction
- **Document classification**: Hierarchical document structure
- **Question answering**: Reasoning over knowledge graphs

### 5. Recommender Systems
- **User-item graphs**: Modeling user-item interactions
- **Session-based recommendation**: Sequential user behavior
- **Multi-modal recommendation**: Combining different data types

## Research Frontiers

### 1. Scalable GNNs
- **Graph sampling**: Efficient sampling for large graphs
- **Distributed training**: Training across multiple machines
- **Approximation methods**: Approximating graph convolutions

### 2. Dynamic GNNs
- **Temporal graphs**: Modeling evolving graph structures
- **Continuous-time GNNs**: Continuous-time dynamics
- **Event-based GNNs**: Event-driven graph updates

### 3. Heterogeneous GNNs
- **Multi-relational graphs**: Multiple edge types
- **Multi-modal graphs**: Different node and edge modalities
- **Knowledge graph completion**: Missing link prediction

### 4. Graph Generation
- **GraphVAE**: Variational autoencoders for graphs
- **GraphGAN**: Generative adversarial networks for graphs
- **Autoregressive generation**: Sequential graph generation

## Usage Examples

### Basic Node Classification
```python
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# Define GNN model
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Create model and data
model = GCN(num_features=1433, num_classes=7)
data = Data(x=torch.randn(2708, 1433), 
            edge_index=torch.randint(0, 2708, (2, 10556)),
            y=torch.randint(0, 7, (2708,)))

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = F.nll_loss

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    
    out = model(data.x, data.edge_index)
    loss = criterion(out, data.y)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
```

### Graph Classification
```python
from torch_geometric.nn import global_mean_pool, GCNConv

class GraphClassifier(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 64)
        self.classifier = torch.nn.Linear(64, num_classes)
    
    def forward(self, x, edge_index, batch):
        # Graph convolutions
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classification
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

# Usage with batch of graphs
model = GraphClassifier(num_features=7, num_classes=2)
batch_data = [Data(x=torch.randn(10, 7), edge_index=torch.randint(0, 10, (2, 20))) 
              for _ in range(4)]

# Process batch
batch = torch_geometric.data.Batch.from_data_list(batch_data)
out = model(batch.x, batch.edge_index, batch.batch)
```

### Link Prediction
```python
class LinkPredictor(torch.nn.Module):
    def __init__(self, num_features, hidden_dim):
        super().__init__()
        self.encoder = GCN(num_features, hidden_dim)
        self.predictor = torch.nn.Linear(2 * hidden_dim, 1)
    
    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)
    
    def decode(self, z, edge_index):
        # Get node embeddings for edge endpoints
        row, col = edge_index
        z_i = z[row]
        z_j = z[col]
        
        # Concatenate and predict
        return torch.sigmoid(self.predictor(torch.cat([z_i, z_j], dim=1)))
    
    def forward(self, x, edge_index, pos_edge_index, neg_edge_index):
        z = self.encode(x, edge_index)
        
        pos_out = self.decode(z, pos_edge_index)
        neg_out = self.decode(z, neg_edge_index)
        
        return pos_out, neg_out

# Training link prediction
model = LinkPredictor(num_features=1433, hidden_dim=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    
    pos_out, neg_out = model(data.x, data.edge_index, 
                            pos_edge_index, neg_edge_index)
    
    pos_loss = -torch.log(pos_out + 1e-15).mean()
    neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
    loss = pos_loss + neg_loss
    
    loss.backward()
    optimizer.step()
```

### Heterogeneous Graph Neural Network
```python
import torch_geometric.nn as gnn
from torch_geometric.data import HeteroData

class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = gnn.HeteroConv({
            ('user', 'follows', 'user'): gnn.GCNConv(-1, hidden_dim),
            ('user', 'likes', 'item'): gnn.GCNConv(-1, hidden_dim),
            ('item', 'rev_likes', 'user'): gnn.GCNConv(-1, hidden_dim),
        })
        self.conv2 = gnn.HeteroConv({
            ('user', 'follows', 'user'): gnn.GCNConv(hidden_dim, out_dim),
            ('user', 'likes', 'item'): gnn.GCNConv(hidden_dim, out_dim),
            ('item', 'rev_likes', 'user'): gnn.GCNConv(hidden_dim, out_dim),
        })
    
    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict

# Create heterogeneous graph
data = HeteroData()
data['user'].x = torch.randn(100, 64)
data['item'].x = torch.randn(50, 64)
data['user', 'follows', 'user'].edge_index = torch.randint(0, 100, (2, 200))
data['user', 'likes', 'item'].edge_index = torch.randint(0, 100, (2, 300))

# Apply model
model = HeteroGNN(hidden_dim=32, out_dim=16)
out_dict = model(data.x_dict, data.edge_index_dict)
```

## Files in this Directory
- `graph_convolution.py`: Graph convolutional layers
- `graph_attention.py`: Attention mechanisms for graphs
- `message_passing.py`: Message passing framework
- `graph_neural_network.py`: Complete GNN architectures
- `example_usage.py`: Working examples
