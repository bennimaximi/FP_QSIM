"""Utilities for obtaining reference statevectors from Qiskit circuits."""

import numpy as np
import qiskit
from qiskit.quantum_info import Statevector


def mocked_statevector(qc: qiskit.QuantumCircuit) -> np.ndarray:
    """Calculate the exact statevector of a quantum circuit using Qiskit.

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
    qc_eval = qc.copy()

    # Avoid triggering layout warnings on transpiled circuits when there are
    # no measurements to remove.
    has_measure = any(instruction.operation.name == "measure" for instruction in qc_eval.data)
    if has_measure:
        qc_eval.remove_final_measurements()

    state_vector = Statevector(qc_eval)
    return np.reshape(np.asarray(state_vector), (2**qc.num_qubits,))
