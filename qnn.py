# qnn_example.py
"""
Quantum Neural Network (QNN) using Qiskit's TwoLayerQNN.
This example integrates a parameterized quantum circuit into a PyTorch model for binary classification.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from qiskit import Aer
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit_machine_learning.neural_networks import TwoLayerQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.utils import QuantumInstance

def create_quantum_circuit(num_qubits=2):
    """
    Constructs a parameterized quantum circuit for QNN.
    
    Args:
        num_qubits (int): Number of qubits.
    
    Returns:
        QuantumCircuit: Parameterized circuit.
    """
    params = ParameterVector('θ', length=num_qubits * 2)
    qc = QuantumCircuit(num_qubits)
    # Feature encoding with RX gates using the first set of parameters.
    for i in range(num_qubits):
        qc.rx(params[i], i)
    # Variational layer with RY gates using the second set of parameters.
    for i in range(num_qubits):
        qc.ry(params[num_qubits + i], i)
    return qc

def build_qnn_model(num_qubits=2):
    """
    Builds a QNN model using TwoLayerQNN and integrates it with PyTorch via TorchConnector.
    
    Args:
        num_qubits (int): Number of qubits.
    
    Returns:
        TorchConnector: A PyTorch compatible model.
    """
    quantum_instance = QuantumInstance(Aer.get_backend('aer_simulator_statevector'))
    qnn = TwoLayerQNN(num_qubits=num_qubits,
                      feature_map=None,
                      ansatz=create_quantum_circuit(num_qubits),
                      quantum_instance=quantum_instance)
    model = TorchConnector(qnn)
    return model

def train_qnn(model, epochs=20):
    # Dummy dataset: 4 samples with 2 features each.
    X = torch.tensor([[0.1, 0.2],
                      [0.8, 0.9],
                      [0.15, 0.15],
                      [0.85, 0.85]], dtype=torch.float32)
    y = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    return model

if __name__ == '__main__':
    qnn_model = build_qnn_model(num_qubits=2)
    trained_model = train_qnn(qnn_model)
    # Print final model outputs
    X_test = torch.tensor([[0.2, 0.3],
                            [0.7, 0.8]], dtype=torch.float32)
    print("QNN model predictions:")
    print(trained_model(X_test).detach().numpy())
