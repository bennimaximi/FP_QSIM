# mocked simulator for testing purposes
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from dataclasses import dataclass
import numpy as np
from qiskit.quantum_info import Operator


@dataclass
class MockSimulator(AerSimulator):
	def __init__(self) -> None:
		super().__init__()
		self._statevector_simulator = AerSimulator(method='statevector')

	def run(self, circuits: QuantumCircuit, shots: int) -> object:  # ty:ignore[invalid-method-override]
		result = self._statevector_simulator.run(circuits, shots=shots).result()
		return result


@dataclass
class CustomSimulator(AerSimulator):
	def run(self, circuit: QuantumCircuit, shots: int) -> object:  # ty:ignore[invalid-method-override]
		"""
		Simulates the circuit by manually applying gate matrices (h and u).
		"""
		n_qubits = circuit.num_qubits
		statevector = np.zeros(2**n_qubits, dtype=complex)
		statevector[0] = 1.0  # Start in the |0...0⟩ state
		for instruction in circuit.data:
			gate = instruction[0]
			qubits = [circuit.find_bit(q).index for q in instruction.qubits]
			gate_matrix = Operator(gate).data
			n_gate_qubits = len(qubits)
			gate_matrix = gate_matrix.reshape([2] * (2 * n_gate_qubits))
			input_indices = list(range(n_qubits))
			output_indices = input_indices.copy()

			# New indices for the gate's output
			gate_out = list(range(n_qubits, n_qubits + n_gate_qubits))
			for i, q_idx in enumerate(qubits):
				output_indices[q_idx] = gate_out[i]
		return 0
