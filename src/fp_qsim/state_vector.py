from qiskit.quantum_info import Statevector
import qiskit
import numpy as np


def mocked_statevector(qc: qiskit.QuantumCircuit) -> np.ndarray:
    """Calculates the exact statevector of a quantum circuit using Qiskit.

    This function prepares the circuit for statevector evaluation by removing
    any final measurement operations, which would otherwise collapse the state
    or cause errors in statevector simulators. It then extracts the dense
    state array and ensures it is flattened.

    Args:
        qc (qiskit.QuantumCircuit): The quantum circuit to evaluate.

    Returns:
        np.ndarray: A 1D complex numpy array of length 2^n (where n is the
            number of qubits in the circuit), representing the final state
            vector of the quantum circuit.
    """
    qc.remove_final_measurements()
    state_vector = Statevector(qc)
    state_vector = np.reshape(np.asarray(state_vector), (2**qc.num_qubits,))
    return state_vector
