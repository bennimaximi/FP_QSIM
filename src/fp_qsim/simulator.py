# mocked simulator for testing purposes
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from dataclasses import dataclass


@dataclass
class MockSimulator(AerSimulator):
	def __init__(self) -> None:
		super().__init__()
		self._statevector_simulator = AerSimulator(method='statevector')

	def run(self, circuits: QuantumCircuit, shots: int) -> object:  # ty:ignore[invalid-method-override]
		result = self._statevector_simulator.run(circuits, shots=shots).result()
		return result
