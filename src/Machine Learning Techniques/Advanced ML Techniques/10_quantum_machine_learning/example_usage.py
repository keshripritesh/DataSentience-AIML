"""
Example Usage: Quantum Machine Learning

This file demonstrates practical usage of quantum machine learning components
including quantum circuits, variational algorithms, and hybrid quantum-classical models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from quantum_neural_networks import (
    QuantumCircuit, HadamardGate, PauliXGate, RotationGate,
    VariationalQuantumCircuit, QuantumFeatureMap, QuantumMeasurement,
    QuantumNeuralNetwork, QuantumSupportVectorMachine, QuantumKernel,
    QuantumVariationalClassifier
)


def example_1_basic_quantum_circuits():
    """Example 1: Basic quantum circuits and gates"""
    print("=" * 60)
    print("Example 1: Basic Quantum Circuits and Gates")
    print("=" * 60)
    
    # Create a simple quantum circuit
    circuit = QuantumCircuit(num_qubits=2)
    
    # Add gates to create a Bell state
    h_gate = HadamardGate()
    x_gate = PauliXGate()
    
    circuit.add_gate(h_gate, 0)  # Apply Hadamard to first qubit
    circuit.add_gate(x_gate, 1)  # Apply X gate to second qubit
    
    # Execute the circuit
    final_state = circuit.execute()
    print(f"Final quantum state: {final_state}")
    
    # Calculate measurement probabilities
    probs = torch.abs(final_state) ** 2
    print(f"Measurement probabilities: {probs}")
    
    # Test different rotation gates
    print("\nTesting rotation gates:")
    initial_state = torch.tensor([1.0, 0.0], dtype=torch.complex64)
    
    for angle in [0, np.pi/4, np.pi/2, np.pi]:
        rx_gate = RotationGate('x', angle)
        result = rx_gate.apply(initial_state)
        print(f"R_x({angle:.2f}): {result}")


def example_2_variational_quantum_circuits():
    """Example 2: Variational quantum circuits with optimization"""
    print("\n" + "=" * 60)
    print("Example 2: Variational Quantum Circuits")
    print("=" * 60)
    
    # Create variational quantum circuit
    vqc = VariationalQuantumCircuit(num_qubits=2, num_layers=3)
    vqc.build_circuit()
    
    print(f"Number of parameters: {len(vqc.parameters)}")
    print(f"Parameter values: {vqc.parameters}")
    
    # Define a simple objective function (maximize |⟨ψ|1⟩|²)
    def objective_function(circuit):
        final_state = circuit.execute()
        # Target state |1⟩ for first qubit
        target_state = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.complex64)
        overlap = torch.abs(torch.conj(final_state) @ target_state) ** 2
        return overlap
    
    # Optimize the circuit parameters
    optimizer = optim.Adam([vqc.parameters], lr=0.1)
    
    print("\nOptimizing circuit parameters...")
    for epoch in range(50):
        optimizer.zero_grad()
        
        # Compute objective
        objective = objective_function(vqc)
        loss = -objective  # Minimize negative objective
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Objective = {objective.item():.4f}")
    
    # Final result
    final_objective = objective_function(vqc)
    print(f"Final objective: {final_objective.item():.4f}")


def example_3_quantum_feature_maps():
    """Example 3: Quantum feature maps for data encoding"""
    print("\n" + "=" * 60)
    print("Example 3: Quantum Feature Maps")
    print("=" * 60)
    
    # Create quantum feature map
    feature_map = QuantumFeatureMap(num_qubits=3, encoding_type="angle")
    
    # Test with different data
    test_data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    print(f"Input data: {test_data}")
    
    # Encode data into quantum state
    quantum_state = feature_map.encode(test_data)
    print(f"Encoded quantum state: {quantum_state}")
    
    # Test amplitude encoding
    amplitude_map = QuantumFeatureMap(num_qubits=2, encoding_type="amplitude")
    amplitude_state = amplitude_map.encode(test_data)
    print(f"Amplitude encoded state: {amplitude_state}")
    
    # Verify normalization
    norm = torch.norm(amplitude_state)
    print(f"State norm: {norm:.6f}")


def example_4_quantum_neural_network():
    """Example 4: Quantum-classical hybrid neural network"""
    print("\n" + "=" * 60)
    print("Example 4: Quantum-Classical Hybrid Neural Network")
    print("=" * 60)
    
    # Create quantum neural network
    qnn = QuantumNeuralNetwork(
        input_size=4,
        hidden_size=8,
        output_size=2,
        num_qubits=4,
        num_layers=2
    )
    
    # Create synthetic dataset
    torch.manual_seed(42)
    X = torch.randn(100, 4)
    y = torch.randint(0, 2, (100,))
    
    # Convert to one-hot encoding
    y_onehot = torch.zeros(100, 2)
    y_onehot.scatter_(1, y.unsqueeze(1), 1)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(qnn.parameters(), lr=0.01)
    
    print("Training quantum neural network...")
    losses = []
    
    for epoch in range(20):
        optimizer.zero_grad()
        
        # Forward pass
        outputs = qnn(X)
        loss = criterion(outputs, y_onehot)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    
    # Plot training loss
    plt.figure(figsize=(8, 6))
    plt.plot(losses)
    plt.title('Training Loss - Quantum Neural Network')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('quantum_nn_training.png')
    plt.close()
    
    print(f"Final loss: {losses[-1]:.4f}")


def example_5_quantum_support_vector_machine():
    """Example 5: Quantum Support Vector Machine"""
    print("\n" + "=" * 60)
    print("Example 5: Quantum Support Vector Machine")
    print("=" * 60)
    
    # Create synthetic dataset
    torch.manual_seed(42)
    n_samples = 50
    
    # Create two classes with different distributions
    class1 = torch.randn(n_samples, 2) + torch.tensor([2.0, 2.0])
    class2 = torch.randn(n_samples, 2) + torch.tensor([-2.0, -2.0])
    
    X = torch.cat([class1, class2], dim=0)
    y = torch.cat([torch.ones(n_samples), -torch.ones(n_samples)])
    
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: {torch.bincount((y + 1).long())}")
    
    # Create and train quantum SVM
    qsvm = QuantumSupportVectorMachine(num_qubits=3, C=1.0)
    qsvm.fit(X, y)
    
    # Make predictions
    predictions = qsvm.predict(X)
    
    # Calculate accuracy
    accuracy = torch.mean((predictions > 0).float() == (y > 0).float())
    print(f"Training accuracy: {accuracy.item():.4f}")
    
    # Test on new data
    X_test = torch.randn(20, 2)
    test_predictions = qsvm.predict(X_test)
    print(f"Test predictions: {test_predictions}")


def example_6_quantum_kernels():
    """Example 6: Quantum kernels for kernel methods"""
    print("\n" + "=" * 60)
    print("Example 6: Quantum Kernels")
    print("=" * 60)
    
    # Create quantum feature map and kernel
    feature_map = QuantumFeatureMap(num_qubits=4, encoding_type="angle")
    quantum_kernel = QuantumKernel(num_qubits=4, feature_map=feature_map)
    
    # Test kernel computation
    x1 = torch.tensor([1.0, 2.0, 3.0, 4.0])
    x2 = torch.tensor([2.0, 3.0, 4.0, 5.0])
    
    kernel_value = quantum_kernel.compute_kernel(x1, x2)
    print(f"Kernel value between x1 and x2: {kernel_value:.6f}")
    
    # Compute kernel matrix for a small dataset
    X = torch.randn(10, 4)
    kernel_matrix = quantum_kernel.compute_kernel_matrix(X)
    
    print(f"Kernel matrix shape: {kernel_matrix.shape}")
    print(f"Kernel matrix diagonal: {torch.diag(kernel_matrix)}")
    
    # Verify kernel properties
    print(f"Kernel is symmetric: {torch.allclose(kernel_matrix, kernel_matrix.T)}")
    print(f"Kernel is positive definite: {torch.all(torch.linalg.eigvals(kernel_matrix).real > -1e-10)}")


def example_7_quantum_measurements():
    """Example 7: Quantum measurements and observables"""
    print("\n" + "=" * 60)
    print("Example 7: Quantum Measurements")
    print("=" * 60)
    
    # Create a quantum state
    state = torch.tensor([0.7071, 0.7071], dtype=torch.complex64)  # (|0⟩ + |1⟩) / √2
    print(f"Quantum state: {state}")
    
    # Create measurement object
    measurement = QuantumMeasurement()
    
    # Measure in computational basis
    probs = measurement.probability_measurement(state)
    print(f"Measurement probabilities: {probs}")
    
    # Test with different observables
    # Pauli-Z observable
    pauli_z = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.complex64)
    expectation_z = measurement.expectation_value(state, pauli_z)
    print(f"⟨Z⟩ expectation value: {expectation_z:.6f}")
    
    # Pauli-X observable
    pauli_x = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex64)
    expectation_x = measurement.expectation_value(state, pauli_x)
    print(f"⟨X⟩ expectation value: {expectation_x:.6f}")
    
    # Identity observable
    identity = torch.eye(2, dtype=torch.complex64)
    expectation_id = measurement.expectation_value(state, identity)
    print(f"⟨I⟩ expectation value: {expectation_id:.6f}")


def example_8_quantum_variational_classifier():
    """Example 8: Quantum variational classifier"""
    print("\n" + "=" * 60)
    print("Example 8: Quantum Variational Classifier")
    print("=" * 60)
    
    # Create quantum variational classifier
    qvc = QuantumVariationalClassifier(
        num_qubits=3,
        num_classes=3,
        num_layers=2
    )
    
    # Test with sample input
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    class_probs = qvc.forward(x)
    
    print(f"Input: {x}")
    print(f"Class probabilities: {class_probs}")
    print(f"Predicted class: {torch.argmax(class_probs).item()}")


def example_9_quantum_entanglement_demo():
    """Example 9: Demonstrating quantum entanglement"""
    print("\n" + "=" * 60)
    print("Example 9: Quantum Entanglement")
    print("=" * 60)
    
    # Create Bell state circuit
    circuit = QuantumCircuit(num_qubits=2)
    
    # Apply Hadamard to first qubit
    h_gate = HadamardGate()
    circuit.add_gate(h_gate, 0)
    
    # Apply CNOT (simulated with X gate for simplicity)
    x_gate = PauliXGate()
    circuit.add_gate(x_gate, 1)
    
    # Execute circuit
    bell_state = circuit.execute()
    print(f"Bell state: {bell_state}")
    
    # Verify entanglement properties
    probs = torch.abs(bell_state) ** 2
    print(f"Measurement probabilities: {probs}")
    
    # Check that only |00⟩ and |11⟩ have non-zero probability
    print(f"Probability of |00⟩: {probs[0]:.6f}")
    print(f"Probability of |01⟩: {probs[1]:.6f}")
    print(f"Probability of |10⟩: {probs[2]:.6f}")
    print(f"Probability of |11⟩: {probs[3]:.6f}")


def example_10_advanced_quantum_circuits():
    """Example 10: Advanced quantum circuit patterns"""
    print("\n" + "=" * 60)
    print("Example 10: Advanced Quantum Circuits")
    print("=" * 60)
    
    # Create a more complex circuit
    circuit = QuantumCircuit(num_qubits=3)
    
    # Apply rotation gates with different angles
    angles = [np.pi/6, np.pi/4, np.pi/3]
    for i, angle in enumerate(angles):
        rx_gate = RotationGate('x', angle)
        ry_gate = RotationGate('y', angle/2)
        rz_gate = RotationGate('z', angle/3)
        
        circuit.add_gate(rx_gate, i)
        circuit.add_gate(ry_gate, i)
        circuit.add_gate(rz_gate, i)
    
    # Execute circuit
    final_state = circuit.execute()
    print(f"Final state: {final_state}")
    
    # Analyze state properties
    probs = torch.abs(final_state) ** 2
    print(f"State norm: {torch.norm(final_state):.6f}")
    print(f"Max probability: {torch.max(probs):.6f}")
    print(f"Min probability: {torch.min(probs):.6f}")
    
    # Find most probable basis state
    max_idx = torch.argmax(probs)
    print(f"Most probable state: |{max_idx:03b}⟩ with probability {probs[max_idx]:.6f}")


def main():
    """Run all quantum machine learning examples"""
    print("Quantum Machine Learning Examples")
    print("=" * 80)
    
    # Run all examples
    example_1_basic_quantum_circuits()
    example_2_variational_quantum_circuits()
    example_3_quantum_feature_maps()
    example_4_quantum_neural_network()
    example_5_quantum_support_vector_machine()
    example_6_quantum_kernels()
    example_7_quantum_measurements()
    example_8_quantum_variational_classifier()
    example_9_quantum_entanglement_demo()
    example_10_advanced_quantum_circuits()
    
    print("\n" + "=" * 80)
    print("All quantum machine learning examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
