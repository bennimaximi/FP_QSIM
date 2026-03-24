# mocked simulator for testing purposes
from qiskit_aer import AerSimulator, StatevectorSimulator
from dataclasses import dataclass

@dataclass
class MockSimulator(AerSimulator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._statevector_simulator = AerSimulator(method='statevector')
        
    
    def run(self, circuits, *args, **kwargs):
        # For testing purposes, we return a statevector instead of a measurement
        result = self._statevector_simulator.run(circuits).result()
        return result

