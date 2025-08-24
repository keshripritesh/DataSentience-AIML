# Few-Shot Learning with Meta-Learning

## Overview
Few-shot learning with meta-learning represents a paradigm shift in machine learning, where models learn to learn rather than learning specific tasks. This approach enables rapid adaptation to new tasks with minimal examples by leveraging knowledge acquired across multiple related tasks during meta-training.

## Core Concepts

### Meta-Learning (Learning to Learn)
Meta-learning, or "learning to learn," is a framework where the learning algorithm itself is learned from data. Instead of training a model to perform a specific task, we train it to quickly adapt to new tasks.

**Key Components:**
- **Meta-Learner**: The algorithm that learns how to learn
- **Base-Learner**: The model that adapts to specific tasks
- **Task Distribution**: The distribution of tasks the meta-learner encounters
- **Inner Loop**: Fast adaptation to a specific task
- **Outer Loop**: Slow learning of the meta-learning algorithm

### Few-Shot Learning
Few-shot learning addresses the challenge of learning new concepts from very few examples (typically 1-5 examples per class).

**Problem Formulation:**
- **N-way K-shot**: N classes with K examples per class
- **Support Set**: Training examples for adaptation
- **Query Set**: Test examples for evaluation
- **Episode**: A single few-shot learning task

### Episode-Based Training
Training is organized into episodes, each representing a few-shot learning task:
```python
# Example episode structure
episode = {
    'support_set': [(x1, y1), (x2, y2), ...],  # Training examples
    'query_set': [(x1', y1'), (x2', y2'), ...], # Test examples
    'n_way': 5,  # Number of classes
    'k_shot': 3  # Examples per class
}
```

## Bizarre and Advanced Aspects

### 1. Learning to Learn
The most bizarre aspect is that the model learns a learning algorithm rather than a specific function. This creates a hierarchy of learning:
- **Level 0**: Learning specific patterns in data
- **Level 1**: Learning how to learn patterns quickly

### 2. Rapid Adaptation
Models can adapt to new tasks in just a few gradient steps, sometimes even a single step. This challenges traditional notions of learning requiring large amounts of data.

### 3. Task-Agnostic Representations
The model learns representations that are useful across many tasks, not just the training tasks. This enables generalization to unseen task distributions.

### 4. Bi-Level Optimization
Meta-learning involves nested optimization problems:
- **Inner optimization**: Fast adaptation to a specific task
- **Outer optimization**: Learning the meta-parameters

### 5. Episodic Training
Training mimics the test-time scenario by creating episodes that simulate few-shot learning tasks.

## Technical Architecture

### MAML (Model-Agnostic Meta-Learning)
MAML learns initial parameters that can be quickly adapted to new tasks.

**Algorithm:**
```python
def maml_algorithm(model, tasks, alpha=0.01, beta=0.001):
    for task in tasks:
        # Inner loop: adapt to task
        adapted_params = model.params - alpha * gradient(task, model.params)
        
        # Outer loop: update meta-parameters
        meta_gradient = gradient(task, adapted_params)
        model.params -= beta * meta_gradient
```

**Key Features:**
- **Gradient-based adaptation**: Uses gradient descent for fast adaptation
- **Task-specific adaptation**: Each task gets its own adapted parameters
- **Meta-parameter learning**: Learns initial parameters that enable fast adaptation

### Prototypical Networks
Prototypical networks learn to classify by computing distances to class prototypes.

**Prototype Computation:**
```python
def compute_prototypes(support_set, n_way, k_shot):
    prototypes = []
    for i in range(n_way):
        class_examples = support_set[i * k_shot:(i + 1) * k_shot]
        prototype = torch.mean(class_examples, dim=0)
        prototypes.append(prototype)
    return torch.stack(prototypes)
```

**Classification:**
```python
def classify_query(query, prototypes):
    distances = euclidean_distance(query, prototypes)
    return softmax(-distances)
```

### Relation Networks
Relation networks learn similarity functions between examples.

**Architecture:**
- **Embedding Module**: Maps inputs to feature representations
- **Relation Module**: Computes similarity between pairs
- **Output**: Similarity score between query and support examples

## Implementation Details

### Episode Generator
```python
class EpisodeGenerator:
    def __init__(self, dataset, n_way, k_shot, query_size):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.query_size = query_size
    
    def generate_episode(self):
        # Sample N classes
        classes = random.sample(self.dataset.classes, self.n_way)
        
        # Sample support and query examples
        support_set = []
        query_set = []
        
        for class_idx in classes:
            class_examples = self.dataset.get_class_examples(class_idx)
            selected = random.sample(class_examples, self.k_shot + self.query_size)
            
            support_set.extend([(ex, class_idx) for ex in selected[:self.k_shot]])
            query_set.extend([(ex, class_idx) for ex in selected[self.k_shot:]])
        
        return support_set, query_set
```

### MAML Implementation
```python
class MAML:
    def __init__(self, model, alpha=0.01, beta=0.001):
        self.model = model
        self.alpha = alpha  # Inner loop learning rate
        self.beta = beta    # Outer loop learning rate
    
    def adapt_to_task(self, support_set, num_steps=5):
        """Fast adaptation to a specific task"""
        adapted_model = copy.deepcopy(self.model)
        
        for _ in range(num_steps):
            loss = self.compute_loss(adapted_model, support_set)
            gradients = torch.autograd.grad(loss, adapted_model.parameters())
            
            # Update parameters
            for param, grad in zip(adapted_model.parameters(), gradients):
                param.data -= self.alpha * grad
        
        return adapted_model
    
    def meta_update(self, tasks):
        """Meta-update across multiple tasks"""
        meta_gradients = []
        
        for task in tasks:
            support_set, query_set = task
            
            # Adapt to task
            adapted_model = self.adapt_to_task(support_set)
            
            # Compute loss on query set
            query_loss = self.compute_loss(adapted_model, query_set)
            
            # Compute gradients with respect to original parameters
            gradients = torch.autograd.grad(query_loss, self.model.parameters())
            meta_gradients.append(gradients)
        
        # Average gradients and update
        avg_gradients = [torch.stack([g[i] for g in meta_gradients]).mean(0) 
                        for i in range(len(meta_gradients[0]))]
        
        for param, grad in zip(self.model.parameters(), avg_gradients):
            param.data -= self.beta * grad
```

## Advanced Variants

### 1. Reptile
A simplified version of MAML that doesn't require second-order gradients.

**Algorithm:**
```python
def reptile_update(model, tasks, epsilon=0.1):
    for task in tasks:
        # Adapt to task
        adapted_params = adapt_to_task(model, task)
        
        # Move towards adapted parameters
        for param, adapted_param in zip(model.parameters(), adapted_params):
            param.data += epsilon * (adapted_param - param)
```

### 2. Prototypical Networks with Attention
Enhances prototypical networks with attention mechanisms.

```python
class AttentionPrototypicalNetworks:
    def __init__(self, embedding_dim, attention_dim):
        self.embedding_net = EmbeddingNetwork(embedding_dim)
        self.attention = AttentionModule(embedding_dim, attention_dim)
    
    def compute_attention_prototypes(self, support_set, n_way, k_shot):
        embeddings = self.embedding_net(support_set)
        prototypes = []
        
        for i in range(n_way):
            class_embeddings = embeddings[i * k_shot:(i + 1) * k_shot]
            attention_weights = self.attention(class_embeddings)
            prototype = torch.sum(attention_weights * class_embeddings, dim=0)
            prototypes.append(prototype)
        
        return torch.stack(prototypes)
```

### 3. Meta-Learning with Memory Networks
Incorporates external memory for better few-shot learning.

```python
class MemoryMetaLearner:
    def __init__(self, memory_size, memory_dim):
        self.memory = torch.randn(memory_size, memory_dim)
        self.meta_learner = MetaLearner()
    
    def adapt_with_memory(self, support_set):
        # Read from memory
        memory_read = self.read_memory(support_set)
        
        # Combine with support set
        enhanced_support = self.combine_with_memory(support_set, memory_read)
        
        # Adapt using enhanced support set
        return self.meta_learner.adapt(enhanced_support)
```

### 4. Gradient-Based Meta-Learning with Uncertainty
Incorporates uncertainty estimation for more robust adaptation.

```python
class UncertaintyMAML:
    def __init__(self, model, uncertainty_estimator):
        self.model = model
        self.uncertainty_estimator = uncertainty_estimator
    
    def adapt_with_uncertainty(self, support_set):
        # Compute uncertainty for each example
        uncertainties = self.uncertainty_estimator(support_set)
        
        # Weight examples by uncertainty
        weighted_support = self.weight_by_uncertainty(support_set, uncertainties)
        
        # Adapt using weighted examples
        return self.adapt_to_task(weighted_support)
```

## Performance Metrics

### 1. Few-Shot Classification Accuracy
- **N-way K-shot accuracy**: Classification accuracy on N-way K-shot tasks
- **Cross-domain accuracy**: Performance on tasks from different domains
- **Adaptation speed**: How quickly the model adapts to new tasks

### 2. Meta-Learning Metrics
- **Meta-training loss**: Loss during the meta-training phase
- **Meta-validation accuracy**: Performance on held-out tasks during meta-training
- **Task adaptation time**: Time required to adapt to new tasks

### 3. Robustness Metrics
- **Domain shift robustness**: Performance under domain shifts
- **Noise robustness**: Performance with noisy support sets
- **Out-of-distribution performance**: Performance on tasks outside the training distribution

## Applications

### 1. Computer Vision
- **Object recognition**: Learning new object categories from few examples
- **Image segmentation**: Adapting to new segmentation tasks
- **Visual question answering**: Learning new question types

### 2. Natural Language Processing
- **Text classification**: Learning new text categories
- **Named entity recognition**: Adapting to new entity types
- **Machine translation**: Learning new language pairs

### 3. Robotics
- **Task adaptation**: Learning new robotic tasks quickly
- **Manipulation skills**: Adapting to new objects and environments
- **Navigation**: Learning new environments with few examples

### 4. Healthcare
- **Medical diagnosis**: Learning new disease categories
- **Drug discovery**: Adapting to new molecular properties
- **Medical imaging**: Learning new imaging modalities

## Research Frontiers

### 1. Meta-Learning for Reinforcement Learning
- **Meta-RL**: Learning to learn reinforcement learning policies
- **Few-shot imitation learning**: Learning from few demonstrations
- **Multi-task RL**: Sharing knowledge across multiple RL tasks

### 2. Neural Architecture Search for Meta-Learning
- **Auto-Meta**: Automatically designing meta-learning architectures
- **Neural architecture search for few-shot learning**: Finding optimal architectures for few-shot tasks

### 3. Uncertainty in Meta-Learning
- **Bayesian meta-learning**: Incorporating uncertainty in meta-learning
- **Calibrated few-shot learning**: Providing reliable uncertainty estimates
- **Active learning in meta-learning**: Selecting informative examples

### 4. Continual Meta-Learning
- **Lifelong meta-learning**: Learning continuously without forgetting
- **Catastrophic forgetting in meta-learning**: Preventing forgetting of previous tasks
- **Incremental meta-learning**: Adding new tasks incrementally

## Usage Examples

### Basic MAML Implementation
```python
import torch
import torch.nn as nn
from maml import MAML
from episode_generator import EpisodeGenerator

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Initialize MAML
model = SimpleModel(input_dim=784, hidden_dim=64, output_dim=5)
maml = MAML(model, alpha=0.01, beta=0.001)

# Generate episodes
episode_generator = EpisodeGenerator(dataset, n_way=5, k_shot=3, query_size=15)

# Meta-training
for epoch in range(1000):
    tasks = [episode_generator.generate_episode() for _ in range(4)]
    maml.meta_update(tasks)
    
    if epoch % 100 == 0:
        # Evaluate on validation tasks
        val_tasks = [episode_generator.generate_episode() for _ in range(10)]
        accuracy = evaluate_maml(maml, val_tasks)
        print(f"Epoch {epoch}, Validation Accuracy: {accuracy:.3f}")
```

### Prototypical Networks
```python
from prototypical_networks import PrototypicalNetworks

# Initialize prototypical networks
proto_net = PrototypicalNetworks(embedding_dim=64)

# Generate episode
support_set, query_set = episode_generator.generate_episode()

# Compute prototypes
prototypes = proto_net.compute_prototypes(support_set, n_way=5, k_shot=3)

# Classify query examples
predictions = proto_net.classify_query(query_set, prototypes)
accuracy = compute_accuracy(predictions, query_labels)
print(f"Prototypical Networks Accuracy: {accuracy:.3f}")
```

### Relation Networks
```python
from relation_networks import RelationNetworks

# Initialize relation networks
relation_net = RelationNetworks(embedding_dim=64, relation_dim=8)

# Generate episode
support_set, query_set = episode_generator.generate_episode()

# Compute relations
relations = relation_net.compute_relations(query_set, support_set)

# Classify based on relations
predictions = relation_net.classify_by_relations(relations)
accuracy = compute_accuracy(predictions, query_labels)
print(f"Relation Networks Accuracy: {accuracy:.3f}")
```

### Advanced: Multi-Modal Few-Shot Learning
```python
class MultiModalFewShotLearner:
    def __init__(self, vision_encoder, text_encoder, fusion_network):
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.fusion_network = fusion_network
    
    def adapt_to_multimodal_task(self, support_set):
        # Encode vision and text modalities
        vision_features = self.vision_encoder(support_set['images'])
        text_features = self.text_encoder(support_set['descriptions'])
        
        # Fuse modalities
        fused_features = self.fusion_network(vision_features, text_features)
        
        # Adapt using fused features
        return self.adapt_to_task(fused_features)

# Usage
multimodal_learner = MultiModalFewShotLearner(
    vision_encoder=VisionEncoder(),
    text_encoder=TextEncoder(),
    fusion_network=FusionNetwork()
)

# Adapt to multimodal task
adapted_model = multimodal_learner.adapt_to_multimodal_task(multimodal_support_set)
```

## Files in this Directory
- `maml.py`: Model-Agnostic Meta-Learning implementation
- `prototypical_networks.py`: Prototypical Networks
- `relation_networks.py`: Relation Networks
- `episode_generator.py`: Episode-based data generation
- `example_usage.py`: Working examples
