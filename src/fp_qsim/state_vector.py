from qiskit.quantum_info import Statevector
import qiskit
import numpy as np


def mocked_statevector(qc: qiskit.QuantumCircuit) -> np.ndarray:
	qc.remove_final_measurements()
	state_vector = Statevector(qc)
	state_vector = np.reshape(np.asarray(state_vector), (2**qc.num_qubits,))
	return state_vector
