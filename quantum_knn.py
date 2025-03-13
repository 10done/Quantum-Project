# quantum_knn.py
"""
Quantum k-Nearest Neighbors (QkNN) using the Swap Test.
This code encodes 2D classical data into single-qubit states and
uses a swap test circuit to estimate the similarity (fidelity) between two states.
The similarity measure is then used as a proxy for the k-NN distance metric.
"""

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Gate

def encode_state(vector):
    """
    Encodes a 2D classical vector into a single-qubit quantum state using amplitude encoding.
    
    Args:
        vector (np.array): A 2D vector.
    
    Returns:
        QuantumCircuit: A circuit that prepares the qubit in the state corresponding to the normalized vector.
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Cannot encode the zero vector.")
    a, b = vector / norm
    # Calculate the rotation angle
    theta = 2 * np.arccos(a)
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)
    return qc

def get_gate_from_circuit(qc):
    """
    Converts a single-qubit state preparation circuit to a Gate object.
    
    Args:
        qc (QuantumCircuit): The circuit to convert.
    
    Returns:
        Gate: A gate representing the circuit.
    """
    return qc.to_gate(label="StatePrep")

def swap_test_circuit(qc1, qc2):
    """
    Constructs a swap test circuit to compare two quantum states.
    
    Args:
        qc1 (QuantumCircuit): Circuit preparing state 1.
        qc2 (QuantumCircuit): Circuit preparing state 2.
    
    Returns:
        QuantumCircuit: A complete swap test circuit.
    """
    # Create a circuit with 3 qubits: one ancilla and two for the states
    qc = QuantumCircuit(3, 1)
    # Apply Hadamard to the ancilla
    qc.h(0)
    # Append state preparation gates to qubits 1 and 2
    gate1 = get_gate_from_circuit(qc1)
    gate2 = get_gate_from_circuit(qc2)
    qc.append(gate1, [1])
    qc.append(gate2, [2])
    # Controlled-SWAP operation with ancilla as control
    qc.cswap(0, 1, 2)
    # Hadamard on the ancilla after swap
    qc.h(0)
    # Measure ancilla
    qc.measure(0, 0)
    return qc

def estimate_similarity(vector1, vector2, shots=1024):
    """
    Estimates the similarity (fidelity) between two vectors using the swap test.
    
    Args:
        vector1, vector2 (np.array): Input 2D vectors.
        shots (int): Number of measurement shots.
    
    Returns:
        float: Estimated similarity value.
    """
    qc1 = encode_state(vector1)
    qc2 = encode_state(vector2)
    swap_qc = swap_test_circuit(qc1, qc2)
    backend = Aer.get_backend('qasm_simulator')
    result = execute(swap_qc, backend=backend, shots=shots).result()
    counts = result.get_counts(swap_qc)
    # Probability of measuring '0' on ancilla gives information on state overlap.
    p0 = counts.get('0', 0) / shots
    similarity = 2 * p0 - 1  # maps fidelity from [0.5,1] to [0,1]
    return similarity

if __name__ == '__main__':
    # Example vectors
    v1 = np.array([0.8, 0.6])
    v2 = np.array([0.7, 0.7])
    sim = estimate_similarity(v1, v2)
    print(f"Estimated similarity between v1 and v2: {sim:.3f}")
