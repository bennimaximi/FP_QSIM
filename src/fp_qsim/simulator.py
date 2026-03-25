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
class CustomSimulator:
	def run(self, circuit: QuantumCircuit, shots: int = 1024) -> np.ndarray:
		"""Apply each gate tensor to the state tensor using numpy.einsum."""
		n_qubits = circuit.num_qubits

		# Tensor axes are ordered as [q_{n-1}, ..., q_0] so flattening matches Qiskit's basis order.
		state = np.zeros([2] * n_qubits, dtype=complex)
		state[(0,) * n_qubits] = 1.0

		for instruction in circuit.data:
			gate = instruction.operation
			qubits = [circuit.find_bit(q).index for q in instruction.qubits][::-1]
			n_gate_qubits = len(qubits)

			# Gate tensor has indices: [out_0, ..., out_k-1, in_0, ..., in_k-1].
			gate_tensor = Operator(gate).data.reshape([2] * (2 * n_gate_qubits))
			# print(Operator(gate).data)

			# Map logical qubits to tensor axes in state.
			state_axes = list(range(n_qubits))[::-1]
			qubit_axes = list(qubits)  # [n_qubits - 1 - q for q in qubits]

			out_axes = state_axes.copy()
			new_gate_out_axes = list(range(n_qubits, n_qubits + n_gate_qubits))
			for i, axis in enumerate(qubit_axes):
				out_axes[n_qubits - 1 - axis] = new_gate_out_axes[i]
			# print(new_gate_out_axes, qubit_axes, state_axes, out_axes)

			# Contract gate inputs with target state axes and keep updated state axes.
			state = np.einsum(
				gate_tensor,
				new_gate_out_axes + qubit_axes,
				state,
				state_axes,
				out_axes,
				optimize=False,
			)

		return state.reshape(-1)
