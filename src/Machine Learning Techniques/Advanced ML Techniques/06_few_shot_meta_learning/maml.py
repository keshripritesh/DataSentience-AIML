"""
Model-Agnostic Meta-Learning (MAML) Implementation
This module implements MAML for few-shot learning with rapid adaptation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List
import copy


class MAMLModel(nn.Module):
    """
    Base model for MAML that can be rapidly adapted to new tasks.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(MAMLModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Classifier
        self.classifier = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits
    
    def get_inner_params(self) -> List[nn.Parameter]:
        """Get parameters that should be updated during inner loop."""
        return list(self.parameters())


class MAML:
    """
    Model-Agnostic Meta-Learning algorithm.
    """
    
    def __init__(self, 
                 model: MAMLModel,
                 inner_lr: float = 0.01,
                 outer_lr: float = 0.001,
                 num_inner_steps: int = 5,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.num_inner_steps = num_inner_steps
        self.device = device
        
        # Outer loop optimizer
        self.outer_optimizer = torch.optim.Adam(self.model.parameters(), lr=outer_lr)
        
    def inner_loop(self, 
                  support_x: torch.Tensor, 
                  support_y: torch.Tensor) -> MAMLModel:
        """
        Perform inner loop adaptation on support set.
        
        Args:
            support_x: Support set inputs [num_support, input_size]
            support_y: Support set labels [num_support]
            
        Returns:
            Adapted model
        """
        # Create a copy of the model for adaptation
        adapted_model = copy.deepcopy(self.model)
        inner_optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        
        # Inner loop adaptation
        for _ in range(self.num_inner_steps):
            # Forward pass
            logits = adapted_model(support_x)
            loss = F.cross_entropy(logits, support_y)
            
            # Backward pass
            inner_optimizer.zero_grad()
            loss.backward()
            inner_optimizer.step()
        
        return adapted_model
    
    def meta_step(self, 
                 support_x: torch.Tensor, 
                 support_y: torch.Tensor,
                 query_x: torch.Tensor, 
                 query_y: torch.Tensor) -> Dict[str, float]:
        """
        Perform a single meta-learning step.
        
        Args:
            support_x: Support set inputs [num_support, input_size]
            support_y: Support set labels [num_support]
            query_x: Query set inputs [num_query, input_size]
            query_y: Query set labels [num_query]
            
        Returns:
            Dictionary containing losses and metrics
        """
        # Inner loop adaptation
        adapted_model = self.inner_loop(support_x, support_y)
        
        # Outer loop evaluation on query set
        query_logits = adapted_model(query_x)
        query_loss = F.cross_entropy(query_logits, query_y)
        
        # Compute accuracy
        predictions = torch.argmax(query_logits, dim=1)
        accuracy = (predictions == query_y).float().mean().item()
        
        # Outer loop update
        self.outer_optimizer.zero_grad()
        query_loss.backward()
        self.outer_optimizer.step()
        
        return {
            'query_loss': query_loss.item(),
            'accuracy': accuracy
        }
    
    def evaluate(self, 
                support_x: torch.Tensor, 
                support_y: torch.Tensor,
                query_x: torch.Tensor, 
                query_y: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate the model on a new task without updating meta-parameters.
        
        Args:
            support_x: Support set inputs [num_support, input_size]
            support_y: Support set labels [num_support]
            query_x: Query set inputs [num_query, input_size]
            query_y: Query set labels [num_query]
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Inner loop adaptation
        adapted_model = self.inner_loop(support_x, support_y)
        
        # Evaluation on query set
        query_logits = adapted_model(query_x)
        query_loss = F.cross_entropy(query_logits, query_y)
        
        # Compute accuracy
        predictions = torch.argmax(query_logits, dim=1)
        accuracy = (predictions == query_y).float().mean().item()
        
        return {
            'query_loss': query_loss.item(),
            'accuracy': accuracy
        }


class PrototypicalNetwork(nn.Module):
    """
    Prototypical Networks for few-shot learning.
    """
    
    def __init__(self, input_size: int, hidden_size: int, embedding_size: int):
        super(PrototypicalNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        
        # Embedding network
        self.embedding_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_size)
        )
        
    def forward(self, 
                support_x: torch.Tensor, 
                support_y: torch.Tensor,
                query_x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through prototypical network.
        
        Args:
            support_x: Support set inputs [num_support, input_size]
            support_y: Support set labels [num_support]
            query_x: Query set inputs [num_query, input_size]
            
        Returns:
            Logits for query set [num_query, num_classes]
        """
        # Get embeddings
        support_embeddings = self.embedding_net(support_x)  # [num_support, embedding_size]
        query_embeddings = self.embedding_net(query_x)      # [num_query, embedding_size]
        
        # Compute prototypes
        unique_labels = torch.unique(support_y)
        prototypes = []
        
        for label in unique_labels:
            # Get embeddings for this class
            class_mask = (support_y == label)
            class_embeddings = support_embeddings[class_mask]
            
            # Compute prototype (mean embedding)
            prototype = class_embeddings.mean(dim=0)  # [embedding_size]
            prototypes.append(prototype)
        
        prototypes = torch.stack(prototypes)  # [num_classes, embedding_size]
        
        # Compute distances to prototypes
        # [num_query, num_classes, embedding_size]
        query_expanded = query_embeddings.unsqueeze(1).expand(-1, len(unique_labels), -1)
        prototypes_expanded = prototypes.unsqueeze(0).expand(query_embeddings.size(0), -1, -1)
        
        # Euclidean distance
        distances = torch.sum((query_expanded - prototypes_expanded) ** 2, dim=2)  # [num_query, num_classes]
        
        # Convert distances to logits (negative distances)
        logits = -distances
        
        return logits


class RelationNetwork(nn.Module):
    """
    Relation Networks for few-shot learning.
    """
    
    def __init__(self, input_size: int, hidden_size: int):
        super(RelationNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Relation module
        self.relation_module = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, 
                support_x: torch.Tensor, 
                support_y: torch.Tensor,
                query_x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through relation network.
        
        Args:
            support_x: Support set inputs [num_support, input_size]
            support_y: Support set labels [num_support]
            query_x: Query set inputs [num_query, input_size]
            
        Returns:
            Relation scores [num_query, num_classes]
        """
        # Extract features
        support_features = self.feature_extractor(support_x)  # [num_support, hidden_size]
        query_features = self.feature_extractor(query_x)      # [num_query, hidden_size]
        
        # Compute prototypes
        unique_labels = torch.unique(support_y)
        prototypes = []
        
        for label in unique_labels:
            class_mask = (support_y == label)
            class_features = support_features[class_mask]
            prototype = class_features.mean(dim=0)  # [hidden_size]
            prototypes.append(prototype)
        
        prototypes = torch.stack(prototypes)  # [num_classes, hidden_size]
        
        # Compute relation scores
        num_query = query_features.size(0)
        num_classes = len(unique_labels)
        
        # Expand for broadcasting
        query_expanded = query_features.unsqueeze(1).expand(-1, num_classes, -1)  # [num_query, num_classes, hidden_size]
        prototypes_expanded = prototypes.unsqueeze(0).expand(num_query, -1, -1)   # [num_query, num_classes, hidden_size]
        
        # Concatenate features
        combined_features = torch.cat([query_expanded, prototypes_expanded], dim=2)  # [num_query, num_classes, hidden_size*2]
        
        # Compute relation scores
        relation_scores = self.relation_module(combined_features).squeeze(-1)  # [num_query, num_classes]
        
        return relation_scores


class EpisodeGenerator:
    """
    Generate episodes for few-shot learning.
    """
    
    def __init__(self, 
                 data: torch.Tensor, 
                 labels: torch.Tensor,
                 n_way: int = 5,
                 k_shot: int = 5,
                 num_query: int = 15):
        self.data = data
        self.labels = labels
        self.n_way = n_way
        self.k_shot = k_shot
        self.num_query = num_query
        
        # Get unique classes
        self.unique_labels = torch.unique(labels)
        
    def generate_episode(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a single episode.
        
        Returns:
            support_x, support_y, query_x, query_y
        """
        # Randomly sample N classes
        selected_classes = torch.randperm(len(self.unique_labels))[:self.n_way]
        
        support_x_list = []
        support_y_list = []
        query_x_list = []
        query_y_list = []
        
        for i, class_idx in enumerate(selected_classes):
            class_label = self.unique_labels[class_idx]
            
            # Get all samples for this class
            class_mask = (self.labels == class_label)
            class_data = self.data[class_mask]
            
            # Randomly shuffle and select samples
            perm = torch.randperm(len(class_data))
            class_data = class_data[perm]
            
            # Split into support and query sets
            support_data = class_data[:self.k_shot]
            query_data = class_data[self.k_shot:self.k_shot + self.num_query]
            
            support_x_list.append(support_data)
            support_y_list.append(torch.full((self.k_shot,), i, dtype=torch.long))
            query_x_list.append(query_data)
            query_y_list.append(torch.full((len(query_data),), i, dtype=torch.long))
        
        # Concatenate all classes
        support_x = torch.cat(support_x_list, dim=0)
        support_y = torch.cat(support_y_list, dim=0)
        query_x = torch.cat(query_x_list, dim=0)
        query_y = torch.cat(query_y_list, dim=0)
        
        return support_x, support_y, query_x, query_y


if __name__ == "__main__":
    # Example usage
    batch_size = 4
    input_size = 10
    hidden_size = 64
    output_size = 5
    n_way = 3
    k_shot = 2
    num_query = 5
    
    # Create dummy data
    num_samples = 100
    data = torch.randn(num_samples, input_size)
    labels = torch.randint(0, 10, (num_samples,))
    
    # Create MAML model
    maml_model = MAMLModel(input_size, hidden_size, output_size)
    maml = MAML(maml_model, inner_lr=0.01, outer_lr=0.001, num_inner_steps=5)
    
    # Create episode generator
    episode_generator = EpisodeGenerator(data, labels, n_way, k_shot, num_query)
    
    # Generate episode
    support_x, support_y, query_x, query_y = episode_generator.generate_episode()
    
    print(f"Support set: {support_x.shape}, {support_y.shape}")
    print(f"Query set: {query_x.shape}, {query_y.shape}")
    
    # Meta-learning step
    results = maml.meta_step(support_x, support_y, query_x, query_y)
    print(f"Meta-learning results: {results}")
    
    # Test prototypical network
    proto_net = PrototypicalNetwork(input_size, hidden_size, hidden_size)
    proto_logits = proto_net(support_x, support_y, query_x)
    print(f"Prototypical network logits: {proto_logits.shape}")
    
    # Test relation network
    relation_net = RelationNetwork(input_size, hidden_size)
    relation_scores = relation_net(support_x, support_y, query_x)
    print(f"Relation network scores: {relation_scores.shape}")
