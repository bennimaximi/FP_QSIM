# mocked simulator for testing purposes
from qiskit_aer import AerSimulator
from dataclasses import dataclass


@dataclass
class MockSimulator(AerSimulator):
	def __init__(self, *args: object, **kwargs: object) -> None:
		super().__init__(*args, **kwargs)
		self._statevector_simulator = AerSimulator(method='statevector')

	def run(self, circuits: object, *args: object, **kwargs: object) -> object:
		# For testing purposes, we return a statevector instead of a measurement
		result = self._statevector_simulator.run(circuits).result()
		return result
