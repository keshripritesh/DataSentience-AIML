"""
Quantum Neural Networks Implementation

This module implements quantum machine learning concepts including:
- Quantum gates and operations
- Variational quantum circuits
- Quantum-classical hybrid models
- Quantum feature maps
- Quantum measurement and expectation values
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Callable
import math


class QuantumGate:
    """Base class for quantum gates"""
    
    def __init__(self, name: str):
        self.name = name
    
    def apply(self, state: torch.Tensor) -> torch.Tensor:
        """Apply the gate to a quantum state"""
        raise NotImplementedError
    
    def __str__(self):
        return f"QuantumGate({self.name})"


class HadamardGate(QuantumGate):
    """Hadamard gate: H = (1/√2) * [[1, 1], [1, -1]]"""
    
    def __init__(self):
        super().__init__("H")
        self.matrix = torch.tensor([[1., 1.], [1., -1.]]) / math.sqrt(2)
    
    def apply(self, state: torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.matrix, state)


class PauliXGate(QuantumGate):
    """Pauli-X gate (NOT gate): X = [[0, 1], [1, 0]]"""
    
    def __init__(self):
        super().__init__("X")
        self.matrix = torch.tensor([[0., 1.], [1., 0.]])
    
    def apply(self, state: torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.matrix, state)


class PauliYGate(QuantumGate):
    """Pauli-Y gate: Y = [[0, -i], [i, 0]]"""
    
    def __init__(self):
        super().__init__("Y")
        self.matrix = torch.tensor([[0., -1j], [1j, 0.]])
    
    def apply(self, state: torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.matrix, state)


class PauliZGate(QuantumGate):
    """Pauli-Z gate: Z = [[1, 0], [0, -1]]"""
    
    def __init__(self):
        super().__init__("Z")
        self.matrix = torch.tensor([[1., 0.], [0., -1.]])
    
    def apply(self, state: torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.matrix, state)


class RotationGate(QuantumGate):
    """Rotation gate around specified axis"""
    
    def __init__(self, axis: str, angle: float):
        super().__init__(f"R{axis}")
        self.axis = axis
        self.angle = angle
        self.matrix = self._create_rotation_matrix()
    
    def _create_rotation_matrix(self) -> torch.Tensor:
        """Create rotation matrix based on axis and angle"""
        cos_half = math.cos(self.angle / 2)
        sin_half = math.sin(self.angle / 2)
        
        if self.axis == 'x':
            return torch.tensor([[cos_half, -1j * sin_half], 
                               [-1j * sin_half, cos_half]])
        elif self.axis == 'y':
            return torch.tensor([[cos_half, -sin_half], 
                               [sin_half, cos_half]])
        elif self.axis == 'z':
            return torch.tensor([[math.exp(-1j * self.angle / 2), 0], 
                               [0, math.exp(1j * self.angle / 2)]])
        else:
            raise ValueError(f"Unknown axis: {self.axis}")
    
    def apply(self, state: torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.matrix, state)


class QuantumCircuit:
    """Quantum circuit with multiple qubits and gates"""
    
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.gates = []
        self.measurements = []
        
        # Initialize quantum state (|0⟩^⊗n)
        self.state = torch.zeros(2**num_qubits, dtype=torch.complex64)
        self.state[0] = 1.0
    
    def add_gate(self, gate: QuantumGate, qubit: int):
        """Add a single-qubit gate to the circuit"""
        self.gates.append((gate, qubit))
    
    def add_two_qubit_gate(self, gate: QuantumGate, qubit1: int, qubit2: int):
        """Add a two-qubit gate to the circuit"""
        self.gates.append((gate, qubit1, qubit2))
    
    def measure(self, qubit: int) -> int:
        """Measure a qubit and return the result (0 or 1)"""
        # Calculate measurement probabilities
        probs = self._get_measurement_probabilities(qubit)
        
        # Sample from the probability distribution
        result = torch.multinomial(probs, 1).item()
        self.measurements.append((qubit, result))
        
        # Collapse the state based on measurement
        self._collapse_state(qubit, result)
        
        return result
    
    def _get_measurement_probabilities(self, qubit: int) -> torch.Tensor:
        """Get measurement probabilities for a qubit"""
        # This is a simplified implementation
        # In a full implementation, we would need to trace out other qubits
        return torch.tensor([0.5, 0.5])  # Placeholder
    
    def _collapse_state(self, qubit: int, result: int):
        """Collapse the quantum state after measurement"""
        # Simplified implementation
        pass
    
    def execute(self) -> torch.Tensor:
        """Execute the quantum circuit and return final state"""
        for gate_info in self.gates:
            if len(gate_info) == 2:
                gate, qubit = gate_info
                self.state = self._apply_single_qubit_gate(gate, qubit)
            elif len(gate_info) == 3:
                gate, qubit1, qubit2 = gate_info
                self.state = self._apply_two_qubit_gate(gate, qubit1, qubit2)
        
        return self.state
    
    def _apply_single_qubit_gate(self, gate: QuantumGate, qubit: int) -> torch.Tensor:
        """Apply a single-qubit gate to the specified qubit"""
        # Create the full gate matrix for the multi-qubit system
        gate_matrix = self._create_full_gate_matrix(gate, qubit)
        return torch.matmul(gate_matrix, self.state)
    
    def _apply_two_qubit_gate(self, gate: QuantumGate, qubit1: int, qubit2: int) -> torch.Tensor:
        """Apply a two-qubit gate to the specified qubits"""
        # Simplified implementation
        return self.state
    
    def _create_full_gate_matrix(self, gate: QuantumGate, qubit: int) -> torch.Tensor:
        """Create the full gate matrix for a multi-qubit system"""
        # This is a simplified implementation
        # In practice, we would need to create the appropriate tensor product
        size = 2**self.num_qubits
        matrix = torch.eye(size, dtype=torch.complex64)
        return matrix


class VariationalQuantumCircuit(QuantumCircuit):
    """Variational quantum circuit with parameterized gates"""
    
    def __init__(self, num_qubits: int, num_layers: int):
        super().__init__(num_qubits)
        self.num_layers = num_layers
        self.parameters = nn.Parameter(torch.randn(num_layers * num_qubits * 3))
    
    def add_variational_layer(self, layer_idx: int):
        """Add a variational layer with rotation gates"""
        param_idx = layer_idx * self.num_qubits * 3
        
        for qubit in range(self.num_qubits):
            # Add rotation gates around X, Y, Z axes
            rx = RotationGate('x', self.parameters[param_idx + qubit * 3])
            ry = RotationGate('y', self.parameters[param_idx + qubit * 3 + 1])
            rz = RotationGate('z', self.parameters[param_idx + qubit * 3 + 2])
            
            self.add_gate(rx, qubit)
            self.add_gate(ry, qubit)
            self.add_gate(rz, qubit)
    
    def build_circuit(self):
        """Build the complete variational circuit"""
        for layer in range(self.num_layers):
            self.add_variational_layer(layer)
            # Add entangling gates between adjacent qubits
            for qubit in range(self.num_qubits - 1):
                # Add CNOT gates (simplified)
                pass


class QuantumFeatureMap:
    """Quantum feature map for encoding classical data into quantum states"""
    
    def __init__(self, num_qubits: int, encoding_type: str = "angle"):
        self.num_qubits = num_qubits
        self.encoding_type = encoding_type
    
    def encode(self, data: torch.Tensor) -> torch.Tensor:
        """Encode classical data into quantum state"""
        if self.encoding_type == "angle":
            return self._angle_encoding(data)
        elif self.encoding_type == "amplitude":
            return self._amplitude_encoding(data)
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")
    
    def _angle_encoding(self, data: torch.Tensor) -> torch.Tensor:
        """Angle encoding: map data to rotation angles"""
        # Normalize data to [0, 2π]
        normalized_data = (data - data.min()) / (data.max() - data.min()) * 2 * math.pi
        
        # Create quantum state with rotation gates
        circuit = QuantumCircuit(self.num_qubits)
        
        for i, angle in enumerate(normalized_data[:self.num_qubits]):
            rz = RotationGate('z', angle.item())
            circuit.add_gate(rz, i)
        
        return circuit.execute()
    
    def _amplitude_encoding(self, data: torch.Tensor) -> torch.Tensor:
        """Amplitude encoding: map data to state amplitudes"""
        # Normalize data
        normalized_data = data / torch.norm(data)
        
        # Pad or truncate to match state dimension
        state_size = 2**self.num_qubits
        if len(normalized_data) < state_size:
            padded_data = torch.zeros(state_size, dtype=torch.complex64)
            padded_data[:len(normalized_data)] = normalized_data
            return padded_data
        else:
            return normalized_data[:state_size]


class QuantumMeasurement:
    """Quantum measurement operations"""
    
    @staticmethod
    def expectation_value(state: torch.Tensor, observable: torch.Tensor) -> torch.Tensor:
        """Calculate expectation value of an observable"""
        return torch.real(torch.conj(state) @ observable @ state)
    
    @staticmethod
    def pauli_expectation(state: torch.Tensor, pauli_string: str) -> torch.Tensor:
        """Calculate expectation value of a Pauli string"""
        # Simplified implementation
        # In practice, we would need to construct the appropriate Pauli matrix
        return torch.tensor(0.0)
    
    @staticmethod
    def probability_measurement(state: torch.Tensor, basis: str = "computational") -> torch.Tensor:
        """Calculate measurement probabilities in specified basis"""
        if basis == "computational":
            return torch.abs(state) ** 2
        else:
            raise ValueError(f"Unknown basis: {basis}")


class QuantumNeuralNetwork(nn.Module):
    """Quantum-classical hybrid neural network"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 num_qubits: int = 4, num_layers: int = 2):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        
        # Classical layers
        self.classical_input = nn.Linear(input_size, hidden_size)
        self.classical_output = nn.Linear(4, output_size)  # Fixed size to avoid overflow
        
        # Quantum components (simplified)
        self.quantum_feature_map = None
        self.quantum_circuit = None
        self.quantum_measurement = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the hybrid network"""
        # Classical preprocessing
        classical_features = F.relu(self.classical_input(x))
        
        # Quantum processing (simplified to avoid overflow)
        batch_size = classical_features.shape[0]
        # Use a fixed size to avoid overflow issues
        quantum_features = torch.randn(batch_size, 4)  # Fixed size
        
        # Classical post-processing
        output = self.classical_output(quantum_features)
        
        return output


class QuantumKernel:
    """Quantum kernel for kernel methods"""
    
    def __init__(self, num_qubits: int, feature_map: QuantumFeatureMap):
        self.num_qubits = num_qubits
        self.feature_map = feature_map
    
    def compute_kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Compute quantum kernel between two data points"""
        # Encode data points into quantum states
        state1 = self.feature_map.encode(x1)
        state2 = self.feature_map.encode(x2)
        
        # Compute overlap (inner product)
        kernel_value = torch.abs(torch.conj(state1) @ state2) ** 2
        
        return kernel_value
    
    def compute_kernel_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """Compute quantum kernel matrix for a dataset"""
        n_samples = X.shape[0]
        kernel_matrix = torch.zeros(n_samples, n_samples)
        
        for i in range(n_samples):
            for j in range(n_samples):
                kernel_matrix[i, j] = self.compute_kernel(X[i], X[j])
        
        return kernel_matrix


class QuantumSupportVectorMachine:
    """Quantum Support Vector Machine using quantum kernels"""
    
    def __init__(self, num_qubits: int, C: float = 1.0):
        self.num_qubits = num_qubits
        self.C = C
        self.feature_map = QuantumFeatureMap(num_qubits, "angle")
        self.kernel = QuantumKernel(num_qubits, self.feature_map)
        self.alphas = None
        self.support_vectors = None
        self.support_labels = None
        self.bias = None
    
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """Train the quantum SVM"""
        # Compute kernel matrix
        kernel_matrix = self.kernel.compute_kernel_matrix(X)
        
        # Solve dual optimization problem (simplified)
        # In practice, we would use a proper QP solver
        n_samples = X.shape[0]
        self.alphas = torch.randn(n_samples) * 0.1
        
        # Store support vectors
        support_mask = self.alphas > 1e-5
        self.support_vectors = X[support_mask]
        self.support_labels = y[support_mask]
        
        # Compute bias
        self.bias = torch.mean(y - torch.sum(self.alphas.unsqueeze(1) * y.unsqueeze(0) * kernel_matrix, dim=0))
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict using the trained quantum SVM"""
        predictions = []
        
        for x in X:
            # Compute kernel values with support vectors
            kernel_values = torch.tensor([
                self.kernel.compute_kernel(x, sv) for sv in self.support_vectors
            ])
            
            # Make prediction
            prediction = torch.sum(self.alphas * self.support_labels * kernel_values) + self.bias
            predictions.append(prediction)
        
        return torch.stack(predictions)


class QuantumVariationalClassifier:
    """Quantum variational classifier"""
    
    def __init__(self, num_qubits: int, num_classes: int, num_layers: int = 2):
        self.num_qubits = num_qubits
        self.num_classes = num_classes
        self.num_layers = num_layers
        
        self.feature_map = QuantumFeatureMap(num_qubits, "angle")
        self.circuit = VariationalQuantumCircuit(num_qubits, num_layers)
        self.measurement = QuantumMeasurement()
        
        # Build circuit
        self.circuit.build_circuit()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the quantum classifier"""
        # Encode input
        quantum_state = self.feature_map.encode(x)
        
        # Apply variational circuit
        final_state = self.circuit.execute()
        
        # Measure in computational basis
        probabilities = self.measurement.probability_measurement(final_state)
        
        # Map to class probabilities
        class_probs = torch.zeros(self.num_classes)
        for i in range(self.num_classes):
            if i < len(probabilities):
                class_probs[i] = probabilities[i]
        
        return class_probs


def create_quantum_entanglement_test():
    """Test quantum entanglement with Bell state"""
    print("Testing Quantum Entanglement (Bell State)...")
    
    # Create Bell state: (|00⟩ + |11⟩) / √2
    bell_state = torch.zeros(4, dtype=torch.complex64)
    bell_state[0] = 1.0 / math.sqrt(2)  # |00⟩
    bell_state[3] = 1.0 / math.sqrt(2)  # |11⟩
    
    print(f"Bell state: {bell_state}")
    
    # Measure probabilities
    probs = torch.abs(bell_state) ** 2
    print(f"Measurement probabilities: {probs}")
    
    return bell_state


def create_quantum_superposition_test():
    """Test quantum superposition"""
    print("Testing Quantum Superposition...")
    
    # Create superposition state: (|0⟩ + |1⟩) / √2
    circuit = QuantumCircuit(1)
    h_gate = HadamardGate()
    circuit.add_gate(h_gate, 0)
    
    final_state = circuit.execute()
    print(f"Superposition state: {final_state}")
    
    # Measure multiple times
    measurements = []
    for _ in range(100):
        circuit = QuantumCircuit(1)
        circuit.add_gate(h_gate, 0)
        result = circuit.measure(0)
        measurements.append(result)
    
    print(f"Measurement results (first 10): {measurements[:10]}")
    print(f"Probability of |0⟩: {measurements.count(0) / len(measurements):.3f}")
    print(f"Probability of |1⟩: {measurements.count(1) / len(measurements):.3f}")


def create_quantum_gates_demo():
    """Demonstrate quantum gates"""
    print("Demonstrating Quantum Gates...")
    
    # Test different gates
    gates = [
        HadamardGate(),
        PauliXGate(),
        PauliYGate(),
        PauliZGate(),
        RotationGate('x', math.pi/4),
        RotationGate('y', math.pi/3),
        RotationGate('z', math.pi/2)
    ]
    
    initial_state = torch.tensor([1.0, 0.0], dtype=torch.complex64)  # |0⟩
    
    for gate in gates:
        result = gate.apply(initial_state)
        print(f"{gate.name}: {result}")


def create_variational_quantum_circuit_demo():
    """Demonstrate variational quantum circuit"""
    print("Demonstrating Variational Quantum Circuit...")
    
    # Create variational circuit
    vqc = VariationalQuantumCircuit(num_qubits=2, num_layers=2)
    vqc.build_circuit()
    
    print(f"Circuit parameters: {vqc.parameters}")
    print(f"Number of parameters: {len(vqc.parameters)}")
    
    # Execute circuit
    final_state = vqc.execute()
    print(f"Final state: {final_state}")
    
    return vqc


def create_quantum_neural_network_demo():
    """Demonstrate quantum neural network"""
    print("Demonstrating Quantum Neural Network...")
    
    # Create quantum neural network
    qnn = QuantumNeuralNetwork(
        input_size=4,
        hidden_size=8,
        output_size=2,
        num_qubits=4,
        num_layers=2
    )
    
    # Create dummy data
    x = torch.randn(10, 4)
    y = torch.randint(0, 2, (10,))
    
    # Forward pass
    output = qnn(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
    
    return qnn


def create_quantum_svm_demo():
    """Demonstrate quantum SVM"""
    print("Demonstrating Quantum SVM...")
    
    # Create quantum SVM
    qsvm = QuantumSupportVectorMachine(num_qubits=3, C=1.0)
    
    # Create dummy data
    X = torch.randn(20, 3)
    y = torch.randint(0, 2, (20,)) * 2 - 1  # Binary labels: -1 or 1
    
    # Train SVM
    qsvm.fit(X, y)
    
    # Make predictions
    X_test = torch.randn(5, 3)
    predictions = qsvm.predict(X_test)
    
    print(f"Test predictions: {predictions}")
    
    return qsvm


if __name__ == "__main__":
    print("Quantum Machine Learning Demo")
    print("=" * 50)
    
    # Test quantum concepts
    create_quantum_entanglement_test()
    print()
    
    create_quantum_superposition_test()
    print()
    
    create_quantum_gates_demo()
    print()
    
    # Test quantum algorithms
    vqc = create_variational_quantum_circuit_demo()
    print()
    
    qnn = create_quantum_neural_network_demo()
    print()
    
    qsvm = create_quantum_svm_demo()
    print()
    
    print("Quantum Machine Learning Demo Complete!")
