# qsvm_example.py
"""
Quantum Support Vector Machine (QSVM) Example.
This script uses Qiskit Machine Learning to implement a QSVM classifier.
A ZZFeatureMap encodes classical data into a quantum feature space,
and the QSVM algorithm classifies synthetic data using a quantum kernel.
"""

import numpy as np
from qiskit import BasicAer
from qiskit.utils import QuantumInstance
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.algorithms import QSVM

def generate_data():
    """
    Generates a simple binary classification dataset.
    
    Returns:
        tuple: (training_data, test_data) as dictionaries mapping labels to numpy arrays.
    """
    training_data = {
        0: np.array([[0.1, 0.2], [0.2, 0.1]]),
        1: np.array([[0.8, 0.9], [0.9, 0.8]])
    }
    test_data = {
        0: np.array([[0.15, 0.15]]),
        1: np.array([[0.85, 0.85]])
    }
    return training_data, test_data

def run_qsvm():
    # Create a feature map with 2 features and 2 repetitions.
    feature_map = ZZFeatureMap(feature_dimension=2, reps=2, entanglement='full')
    
    # Set up the quantum instance for simulation.
    backend = BasicAer.get_backend('qasm_simulator')
    quantum_instance = QuantumInstance(backend, shots=1024)
    
    # Create the quantum kernel using the feature map.
    quantum_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=quantum_instance)
    
    # Generate training and test data.
    training_data, test_data = generate_data()
    
    # Instantiate and run the QSVM algorithm.
    qsvm = QSVM(quantum_kernel=quantum_kernel, training_dataset=training_data, test_dataset=test_data)
    result = qsvm.run()
    
    return result

if __name__ == '__main__':
    result = run_qsvm()
    print("QSVM Classification Results:")
    print(result)
