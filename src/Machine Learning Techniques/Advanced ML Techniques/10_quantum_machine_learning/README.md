# Quantum Machine Learning

## Overview
Quantum Machine Learning (QML) represents a paradigm shift in computational intelligence by harnessing the unique properties of quantum mechanics—superposition, entanglement, and interference—to enhance classical machine learning algorithms and create entirely new quantum-native approaches. This field combines the power of quantum computing with the flexibility of machine learning, potentially offering exponential speedups for certain problems and enabling solutions to classically intractable tasks.

## Core Concepts

### Quantum Superposition
Unlike classical bits that are either 0 or 1, quantum bits (qubits) can exist in a superposition of states:

```
|ψ⟩ = α|0⟩ + β|1⟩
```

Where:
- `|0⟩` and `|1⟩` are the computational basis states
- `α` and `β` are complex amplitudes with |α|² + |β|² = 1
- The qubit is in both states simultaneously until measured

**Key Insight:** Superposition enables quantum computers to process exponentially many states simultaneously.

### Quantum Entanglement
Entanglement creates correlated quantum states that cannot be described independently:

```
|ψ⟩ = (|00⟩ + |11⟩)/√2
```

This Bell state represents two qubits that are perfectly correlated—measuring one immediately determines the state of the other, regardless of distance.

### Quantum Gates
Quantum gates are unitary operations that manipulate quantum states:

**Single-Qubit Gates:**
- **Hadamard (H)**: Creates superposition: H|0⟩ = (|0⟩ + |1⟩)/√2
- **Pauli-X (X)**: Bit flip: X|0⟩ = |1⟩
- **Pauli-Y (Y)**: Phase and bit flip: Y|0⟩ = i|1⟩
- **Pauli-Z (Z)**: Phase flip: Z|0⟩ = |0⟩, Z|1⟩ = -|1⟩
- **Rotation Gates**: R_x(θ), R_y(θ), R_z(θ) for arbitrary rotations

**Multi-Qubit Gates:**
- **CNOT**: Controlled-NOT gate for entanglement
- **SWAP**: Exchanges qubit states
- **Toffoli**: Three-qubit controlled-controlled-NOT

### Variational Quantum Circuits (VQC)
VQCs are parameterized quantum circuits optimized using classical methods:

```
U(θ) = ∏ᵢ Uᵢ(θᵢ)
```

Where:
- `U(θ)` is the parameterized unitary
- `θ` are the variational parameters
- The circuit is optimized to minimize a cost function

### Quantum-Classical Hybrid Models
Hybrid models combine quantum and classical components:

```
f(x) = C(⟨ψ(x, θ)|M|ψ(x, θ)⟩)
```

Where:
- `|ψ(x, θ)⟩` is the quantum state prepared by the circuit
- `M` is a measurement operator
- `C` is a classical post-processing function

## Bizarre and Advanced Aspects

### 1. Quantum Parallelism
Quantum computers can evaluate functions on all possible inputs simultaneously through superposition, enabling exponential speedups for certain algorithms.

### 2. Quantum Interference
Quantum states can interfere constructively or destructively, allowing quantum algorithms to amplify correct solutions and suppress incorrect ones.

### 3. Measurement Collapse
Quantum states collapse to a definite value upon measurement, making quantum algorithms probabilistic and requiring multiple runs for reliable results.

### 4. No-Cloning Theorem
Quantum states cannot be perfectly copied, fundamentally limiting certain quantum operations and requiring careful circuit design.

### 5. Quantum Decoherence
Quantum systems lose their quantum properties when interacting with the environment, limiting the depth of quantum circuits on current hardware.

### 6. Quantum Advantage
QML can potentially solve certain problems exponentially faster than classical methods, particularly in optimization, simulation, and cryptography.

## Technical Architecture

### Quantum Neural Network
```python
import pennylane as qml
import torch
import torch.nn as nn

class QuantumNeuralNetwork(nn.Module):
    def __init__(self, n_qubits, n_layers, n_classes):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes
        
        # Quantum device
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        # Quantum circuit
        @qml.qnode(self.dev)
        def quantum_circuit(inputs, weights):
            # Encode classical data into quantum state
            self.encode_inputs(inputs)
            
            # Apply variational layers
            for layer in range(n_layers):
                self.variational_layer(weights[layer])
            
            # Measure all qubits
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        self.quantum_circuit = quantum_circuit
        
        # Classical post-processing
        self.classical_layer = nn.Linear(n_qubits, n_classes)
        
        # Initialize weights
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3))
    
    def encode_inputs(self, inputs):
        # Angle encoding: map classical data to rotation angles
        for i in range(self.n_qubits):
            qml.RY(inputs[i], wires=i)
            qml.RZ(inputs[i], wires=i)
    
    def variational_layer(self, weights):
        # Apply rotations and entangling gates
        for i in range(self.n_qubits):
            qml.Rot(*weights[i], wires=i)
        
        # Entangle qubits
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[self.n_qubits - 1, 0])
    
    def forward(self, x):
        # Quantum part
        quantum_output = self.quantum_circuit(x, self.weights)
        quantum_output = torch.tensor(quantum_output, dtype=torch.float32)
        
        # Classical part
        output = self.classical_layer(quantum_output)
        return output
```

### Variational Quantum Eigensolver (VQE)
```python
class VQE:
    def __init__(self, n_qubits, ansatz_type="hardware_efficient"):
        self.n_qubits = n_qubits
        self.ansatz_type = ansatz_type
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(self.dev)
        def circuit(params):
            self.ansatz(params)
            return qml.expval(qml.Hamiltonian(coeffs, observables))
        
        self.circuit = circuit
    
    def ansatz(self, params):
        if self.ansatz_type == "hardware_efficient":
            self.hardware_efficient_ansatz(params)
        elif self.ansatz_type == "unitary_coupled_cluster":
            self.ucc_ansatz(params)
    
    def hardware_efficient_ansatz(self, params):
        # Hardware-efficient ansatz
        for i in range(self.n_qubits):
            qml.RY(params[i], wires=i)
            qml.RZ(params[i + self.n_qubits], wires=i)
        
        # Entangling layer
        for i in range(0, self.n_qubits - 1, 2):
            qml.CNOT(wires=[i, i + 1])
        
        for i in range(1, self.n_qubits - 1, 2):
            qml.CNOT(wires=[i, i + 1])
    
    def ucc_ansatz(self, params):
        # Unitary Coupled Cluster ansatz
        # Apply excitation operators
        for i, param in enumerate(params):
            qml.RX(param, wires=i % self.n_qubits)
            qml.RY(param, wires=(i + 1) % self.n_qubits)
            qml.CNOT(wires=[i % self.n_qubits, (i + 1) % self.n_qubits])
    
    def optimize(self, initial_params, optimizer):
        params = initial_params
        
        for iteration in range(100):
            # Compute energy
            energy = self.circuit(params)
            
            # Update parameters
            params = optimizer.step(self.circuit, params)
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Energy: {energy:.6f}")
        
        return params
```

### Quantum Support Vector Machine (QSVM)
```python
class QuantumSVM:
    def __init__(self, n_qubits, feature_map="ZZFeatureMap"):
        self.n_qubits = n_qubits
        self.feature_map = feature_map
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(self.dev)
        def kernel_circuit(x1, x2):
            self.quantum_feature_map(x1)
            qml.adjoint(self.quantum_feature_map)(x2)
            return qml.probs(wires=range(n_qubits))
        
        self.kernel_circuit = kernel_circuit
    
    def quantum_feature_map(self, x):
        if self.feature_map == "ZZFeatureMap":
            self.zz_feature_map(x)
        elif self.feature_map == "PauliFeatureMap":
            self.pauli_feature_map(x)
    
    def zz_feature_map(self, x):
        # ZZ feature map
        for i in range(self.n_qubits):
            qml.Hadamard(wires=i)
            qml.RZ(x[i], wires=i)
        
        # Entangling layer
        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                qml.CRZ(x[i] * x[j], wires=[i, j])
    
    def pauli_feature_map(self, x):
        # Pauli feature map
        for i in range(self.n_qubits):
            qml.Hadamard(wires=i)
            qml.RZ(x[i], wires=i)
            qml.RX(x[i], wires=i)
    
    def quantum_kernel(self, x1, x2):
        # Compute quantum kernel
        probs = self.kernel_circuit(x1, x2)
        return probs[0]  # Probability of measuring |0...0⟩
    
    def fit(self, X, y):
        # Compute kernel matrix
        n_samples = len(X)
        K = torch.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.quantum_kernel(X[i], X[j])
        
        # Solve SVM optimization problem
        self.alphas = self.solve_svm(K, y)
        self.support_vectors = X
        self.support_labels = y
    
    def predict(self, X):
        predictions = []
        for x in X:
            pred = 0
            for i, (sv, label) in enumerate(zip(self.support_vectors, self.support_labels)):
                pred += self.alphas[i] * label * self.quantum_kernel(x, sv)
            predictions.append(torch.sign(pred))
        return torch.tensor(predictions)
```

## Implementation Details

### Quantum Feature Maps
```python
class QuantumFeatureMap:
    def __init__(self, n_qubits, encoding_type="angle"):
        self.n_qubits = n_qubits
        self.encoding_type = encoding_type
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(self.dev)
        def feature_map_circuit(x):
            self.encode(x)
            return qml.state()
        
        self.circuit = feature_map_circuit
    
    def encode(self, x):
        if self.encoding_type == "angle":
            self.angle_encoding(x)
        elif self.encoding_type == "amplitude":
            self.amplitude_encoding(x)
        elif self.encoding_type == "basis":
            self.basis_encoding(x)
    
    def angle_encoding(self, x):
        # Map classical data to rotation angles
        for i in range(min(len(x), self.n_qubits)):
            qml.RY(x[i], wires=i)
            qml.RZ(x[i], wires=i)
    
    def amplitude_encoding(self, x):
        # Encode data in amplitudes (requires 2^n qubits for n features)
        # Normalize input
        x_norm = x / torch.norm(x)
        
        # Apply rotations to prepare state
        for i in range(self.n_qubits):
            qml.RY(torch.arccos(x_norm[i]), wires=i)
    
    def basis_encoding(self, x):
        # Encode binary data in computational basis
        for i in range(min(len(x), self.n_qubits)):
            if x[i] > 0.5:
                qml.PauliX(wires=i)
    
    def get_feature_vector(self, x):
        # Get quantum feature vector
        state = self.circuit(x)
        return state
```

### Quantum Generative Models
```python
class QuantumGenerativeAdversarialNetwork:
    def __init__(self, n_qubits, latent_dim):
        self.n_qubits = n_qubits
        self.latent_dim = latent_dim
        
        # Quantum generator
        self.generator = QuantumGenerator(n_qubits, latent_dim)
        
        # Classical discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(n_qubits, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def generate_samples(self, n_samples):
        # Generate quantum samples
        latent_noise = torch.randn(n_samples, self.latent_dim)
        quantum_samples = self.generator(latent_noise)
        return quantum_samples
    
    def train_step(self, real_data, n_fake_samples=32):
        # Train discriminator
        fake_samples = self.generate_samples(n_fake_samples)
        
        real_labels = torch.ones(real_data.shape[0], 1)
        fake_labels = torch.zeros(n_fake_samples, 1)
        
        d_real = self.discriminator(real_data)
        d_fake = self.discriminator(fake_samples.detach())
        
        d_loss = F.binary_cross_entropy(d_real, real_labels) + \
                 F.binary_cross_entropy(d_fake, fake_labels)
        
        # Train generator
        fake_samples = self.generate_samples(n_fake_samples)
        g_fake = self.discriminator(fake_samples)
        
        g_loss = F.binary_cross_entropy(g_fake, torch.ones(n_fake_samples, 1))
        
        return d_loss, g_loss

class QuantumGenerator(nn.Module):
    def __init__(self, n_qubits, latent_dim):
        super().__init__()
        self.n_qubits = n_qubits
        self.latent_dim = latent_dim
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(self.dev)
        def generator_circuit(latent_input, weights):
            # Encode latent variables
            for i in range(min(self.latent_dim, self.n_qubits)):
                qml.RY(latent_input[i], wires=i)
            
            # Apply variational layers
            for layer in weights:
                self.variational_layer(layer)
            
            # Measure all qubits
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        self.circuit = generator_circuit
        self.weights = nn.Parameter(torch.randn(3, n_qubits, 3))
    
    def variational_layer(self, weights):
        # Apply rotations and entangling gates
        for i in range(self.n_qubits):
            qml.Rot(*weights[i], wires=i)
        
        # Entangle qubits
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
    
    def forward(self, latent_input):
        # Generate quantum samples
        output = self.circuit(latent_input, self.weights)
        return torch.tensor(output, dtype=torch.float32)
```

### Quantum Error Correction
```python
class QuantumErrorCorrection:
    def __init__(self, code_type="bit_flip"):
        self.code_type = code_type
        
        if code_type == "bit_flip":
            self.encode_circuit = self.bit_flip_encode
            self.decode_circuit = self.bit_flip_decode
        elif code_type == "phase_flip":
            self.encode_circuit = self.phase_flip_encode
            self.decode_circuit = self.phase_flip_decode
    
    def bit_flip_encode(self, logical_qubit):
        # Encode one logical qubit into three physical qubits
        qml.CNOT(wires=[logical_qubit, 1])
        qml.CNOT(wires=[logical_qubit, 2])
    
    def bit_flip_decode(self):
        # Detect and correct bit flip errors
        qml.CNOT(wires=[0, 3])  # Ancilla qubit
        qml.CNOT(wires=[1, 3])
        qml.CNOT(wires=[0, 4])  # Second ancilla
        qml.CNOT(wires=[2, 4])
        
        # Measure ancilla qubits to detect errors
        return [qml.measure(3), qml.measure(4)]
    
    def phase_flip_encode(self, logical_qubit):
        # Encode one logical qubit into three physical qubits
        qml.Hadamard(wires=logical_qubit)
        qml.CNOT(wires=[logical_qubit, 1])
        qml.CNOT(wires=[logical_qubit, 2])
        qml.Hadamard(wires=logical_qubit)
        qml.Hadamard(wires=1)
        qml.Hadamard(wires=2)
    
    def phase_flip_decode(self):
        # Detect and correct phase flip errors
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        qml.Hadamard(wires=2)
        qml.CNOT(wires=[0, 3])
        qml.CNOT(wires=[1, 3])
        qml.CNOT(wires=[0, 4])
        qml.CNOT(wires=[2, 4])
        
        return [qml.measure(3), qml.measure(4)]
```

## Advanced Variants

### 1. Quantum Convolutional Neural Networks
```python
class QuantumCNN(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(self.dev)
        def conv_circuit(inputs, weights):
            self.quantum_conv_layer(inputs, weights)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        self.circuit = conv_circuit
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3))
    
    def quantum_conv_layer(self, inputs, weights):
        # Encode inputs
        for i in range(self.n_qubits):
            qml.RY(inputs[i], wires=i)
        
        # Apply convolutional-like operations
        for layer in weights:
            # Local rotations
            for i in range(self.n_qubits):
                qml.Rot(*layer[i], wires=i)
            
            # Local entangling (convolutional pattern)
            for i in range(0, self.n_qubits - 1, 2):
                qml.CNOT(wires=[i, i + 1])
            for i in range(1, self.n_qubits - 1, 2):
                qml.CNOT(wires=[i, i + 1])
    
    def forward(self, x):
        output = self.circuit(x, self.weights)
        return torch.tensor(output, dtype=torch.float32)
```

### 2. Quantum Recurrent Neural Networks
```python
class QuantumRNN(nn.Module):
    def __init__(self, n_qubits, hidden_dim):
        super().__init__()
        self.n_qubits = n_qubits
        self.hidden_dim = hidden_dim
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(self.dev)
        def rnn_circuit(input_state, hidden_state, weights):
            # Initialize with hidden state
            self.encode_state(hidden_state)
            
            # Process input
            self.process_input(input_state, weights)
            
            # Return new hidden state
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        self.circuit = rnn_circuit
        self.weights = nn.Parameter(torch.randn(n_qubits, 3))
    
    def encode_state(self, state):
        for i in range(self.n_qubits):
            qml.RY(state[i], wires=i)
    
    def process_input(self, input_state, weights):
        # Apply input-dependent rotations
        for i in range(self.n_qubits):
            qml.Rot(*weights[i], wires=i)
        
        # Entangle qubits
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
    
    def forward(self, x_sequence):
        hidden_state = torch.zeros(self.n_qubits)
        outputs = []
        
        for x in x_sequence:
            hidden_state = self.circuit(x, hidden_state, self.weights)
            hidden_state = torch.tensor(hidden_state, dtype=torch.float32)
            outputs.append(hidden_state)
        
        return torch.stack(outputs)
```

### 3. Quantum Boltzmann Machines
```python
class QuantumBoltzmannMachine:
    def __init__(self, n_visible, n_hidden):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_qubits = n_visible + n_hidden
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        
        @qml.qnode(self.dev)
        def energy_circuit(state):
            # Prepare state
            for i in range(self.n_qubits):
                if state[i] == 1:
                    qml.PauliX(wires=i)
            
            # Measure energy
            return qml.expval(self.hamiltonian)
        
        self.circuit = energy_circuit
        self.hamiltonian = self.create_hamiltonian()
    
    def create_hamiltonian(self):
        # Create Ising Hamiltonian
        coeffs = []
        observables = []
        
        # Local fields
        for i in range(self.n_qubits):
            coeffs.append(torch.randn(1))
            observables.append(qml.PauliZ(i))
        
        # Interactions
        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                coeffs.append(torch.randn(1))
                observables.append(qml.PauliZ(i) @ qml.PauliZ(j))
        
        return qml.Hamiltonian(coeffs, observables)
    
    def sample(self, n_samples=100):
        # Quantum sampling using quantum annealing
        samples = []
        
        for _ in range(n_samples):
            # Prepare superposition state
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            
            # Measure
            sample = [qml.measure(i) for i in range(self.n_qubits)]
            samples.append(sample)
        
        return torch.tensor(samples)
```

## Performance Metrics

### 1. Quantum Advantage Metrics
- **Speedup**: Ratio of classical to quantum runtime
- **Quantum volume**: Measure of quantum computer capability
- **Circuit depth**: Number of sequential operations
- **Gate count**: Total number of quantum gates

### 2. Accuracy Metrics
- **Classification accuracy**: For quantum classifiers
- **Energy convergence**: For VQE algorithms
- **Fidelity**: Quantum state overlap
- **Success probability**: For probabilistic algorithms

### 3. Hardware Metrics
- **Coherence time**: How long quantum states persist
- **Gate fidelity**: Accuracy of quantum operations
- **Connectivity**: Qubit interaction patterns
- **Error rates**: Probability of errors per operation

## Applications

### 1. Quantum Chemistry
- **Molecular simulation**: Electronic structure calculations
- **Drug discovery**: Protein folding and binding
- **Material science**: Novel material properties
- **Catalysis**: Chemical reaction optimization

### 2. Optimization
- **Combinatorial optimization**: Traveling salesman, scheduling
- **Portfolio optimization**: Financial risk management
- **Logistics**: Supply chain optimization
- **Machine learning**: Hyperparameter optimization

### 3. Cryptography
- **Quantum key distribution**: Secure communication
- **Post-quantum cryptography**: Quantum-resistant algorithms
- **Random number generation**: True randomness
- **Digital signatures**: Quantum signatures

### 4. Machine Learning
- **Quantum neural networks**: Enhanced neural computation
- **Quantum kernels**: Quantum-enhanced SVM
- **Quantum generative models**: Novel data generation
- **Quantum reinforcement learning**: Quantum agents

## Research Frontiers

### 1. Quantum Supremacy
- **Demonstrating quantum advantage**: Proving quantum computers can solve problems faster
- **Quantum algorithms**: Developing new quantum algorithms
- **Error correction**: Scaling quantum computers with error correction
- **Quantum software**: Programming languages and frameworks

### 2. Quantum-Classical Hybrid
- **Hybrid algorithms**: Combining quantum and classical computation
- **Quantum machine learning**: Quantum-enhanced ML algorithms
- **Quantum optimization**: Quantum-enhanced optimization
- **Quantum simulation**: Quantum simulation of quantum systems

### 3. Quantum Internet
- **Quantum networks**: Distributed quantum computing
- **Quantum repeaters**: Long-distance quantum communication
- **Quantum memory**: Storing quantum states
- **Quantum sensors**: Ultra-sensitive measurements

### 4. Quantum Materials
- **Topological quantum computing**: Using topological states
- **Majorana fermions**: Exotic quantum particles
- **Quantum dots**: Artificial atoms for quantum computing
- **Superconducting qubits**: Current leading technology

## Usage Examples

### Basic Quantum Neural Network
```python
import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim

# Create quantum device
dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    # Encode inputs
    qml.RY(inputs[0], wires=0)
    qml.RY(inputs[1], wires=1)
    
    # Apply variational layers
    qml.Rot(*weights[0], wires=0)
    qml.Rot(*weights[1], wires=1)
    
    # Entangle qubits
    qml.CNOT(wires=[0, 1])
    
    # Measure
    return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]

# Training
weights = torch.randn(2, 3, requires_grad=True)
optimizer = optim.Adam([weights], lr=0.1)

for epoch in range(100):
    # Sample data
    inputs = torch.randn(10, 2)
    targets = torch.randint(0, 2, (10, 2)).float()
    
    loss = 0
    for x, y in zip(inputs, targets):
        output = quantum_circuit(x, weights)
        loss += torch.mean((torch.tensor(output) - y) ** 2)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

### Variational Quantum Eigensolver
```python
import pennylane as qml
import numpy as np

# Define Hamiltonian (example: H2 molecule)
coeffs = [0.011280, 0.171201, 0.171201, 0.011280, -0.222796, -0.222796, 0.168336]
observables = [
    qml.Identity(0), qml.PauliZ(0), qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliZ(1),
    qml.PauliX(0) @ qml.PauliX(1), qml.PauliY(0) @ qml.PauliY(1), qml.PauliZ(0) @ qml.PauliZ(1)
]

hamiltonian = qml.Hamiltonian(coeffs, observables)

# VQE circuit
dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def vqe_circuit(params):
    qml.RY(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(hamiltonian)

# Optimization
params = np.array([0.1, 0.1], requires_grad=True)
optimizer = qml.GradientDescentOptimizer(stepsize=0.4)

for iteration in range(100):
    params, energy = optimizer.step_and_cost(vqe_circuit, params)
    
    if iteration % 10 == 0:
        print(f"Iteration {iteration}, Energy: {energy:.6f}")

print(f"Ground state energy: {energy:.6f}")
```

### Quantum Support Vector Machine
```python
import pennylane as qml
import numpy as np
from sklearn.datasets import make_blobs

# Generate data
X, y = make_blobs(n_samples=100, centers=2, random_state=42)
y = 2 * y - 1  # Convert to {-1, 1}

# Quantum kernel
dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def quantum_kernel(x1, x2):
    # Feature map
    qml.RY(x1[0], wires=0)
    qml.RY(x1[1], wires=1)
    qml.CNOT(wires=[0, 1])
    
    # Adjoint of feature map
    qml.adjoint(qml.CNOT)(wires=[0, 1])
    qml.adjoint(qml.RY)(x2[1], wires=1)
    qml.adjoint(qml.RY)(x2[0], wires=0)
    
    return qml.probs(wires=[0, 1])

# Compute kernel matrix
n_samples = len(X)
K = np.zeros((n_samples, n_samples))

for i in range(n_samples):
    for j in range(n_samples):
        probs = quantum_kernel(X[i], X[j])
        K[i, j] = probs[0]  # Probability of |00⟩ state

# SVM optimization (simplified)
from sklearn.svm import SVC
svm = SVC(kernel='precomputed')
svm.fit(K, y)

# Predictions
predictions = svm.predict(K)
accuracy = np.mean(predictions == y)
print(f"QSVM Accuracy: {accuracy:.3f}")
```

### Quantum Generative Model
```python
import pennylane as qml
import torch
import torch.nn as nn

# Quantum generator
dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev)
def quantum_generator(latent_input, weights):
    # Encode latent variables
    for i in range(4):
        qml.RY(latent_input[i], wires=i)
    
    # Apply variational layers
    for layer in weights:
        for i in range(4):
            qml.Rot(*layer[i], wires=i)
        
        # Entangle
        for i in range(3):
            qml.CNOT(wires=[i, i + 1])
    
    # Measure
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]

# Training
weights = torch.randn(3, 4, 3, requires_grad=True)
optimizer = torch.optim.Adam([weights], lr=0.01)

for epoch in range(1000):
    # Generate samples
    latent_noise = torch.randn(32, 4)
    fake_samples = []
    
    for noise in latent_noise:
        sample = quantum_generator(noise, weights)
        fake_samples.append(torch.tensor(sample))
    
    fake_samples = torch.stack(fake_samples)
    
    # Simple loss (example: maximize variance)
    loss = -torch.var(fake_samples)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Generate new samples
with torch.no_grad():
    new_noise = torch.randn(10, 4)
    new_samples = []
    for noise in new_noise:
        sample = quantum_generator(noise, weights)
        new_samples.append(torch.tensor(sample))
    
    print("Generated samples:")
    for i, sample in enumerate(new_samples):
        print(f"Sample {i}: {sample}")
```

## Files in this Directory
- `quantum_circuits.py`: Quantum circuit implementations
- `quantum_neural_networks.py`: Quantum neural networks
- `variational_quantum_circuits.py`: VQE and VQC
- `quantum_simulator.py`: Quantum state simulation
- `example_usage.py`: Working examples
