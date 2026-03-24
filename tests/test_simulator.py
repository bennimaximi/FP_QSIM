# pytest code for simulator.py
# import pytest
from qiskit import QuantumCircuit
from fp_qsim.simulator import MockSimulator
from qiskit_aer import AerSimulator
import numpy as np


def reference_simulator() -> AerSimulator:
	return AerSimulator(method='statevector')


def mock_simulator() -> MockSimulator:
	return MockSimulator()


def test_bell_state() -> None:
	# Create a Bell state circuit
	qc = QuantumCircuit(2)
	qc.h(0)
	qc.cx(0, 1)

	qc.save_statevector()  # type: ignore

	# Get results from both simulators
	ref_sim = reference_simulator()
	mock_sim = mock_simulator()

	ref_result = ref_sim.run(qc).result()
	mock_result = mock_sim.run(qc, shots=1024)

	# Extract statevectors
	ref_statevector = ref_result.get_statevector()
	mock_statevector = mock_result.get_statevector()  # type: ignore

	# Assert that the statevectors are approximately equal
	assert np.allclose(ref_statevector, mock_statevector)


def test_ghz_state() -> None:
	qc = QuantumCircuit(3)
	qc.h(0)
	qc.cx(0, 1)
	qc.cx(1, 2)

	qc.save_statevector()  # type: ignore

	ref_sim = reference_simulator()
	mock_sim = mock_simulator()

	ref_result = ref_sim.run(qc).result()
	mock_result = mock_sim.run(qc, shots=1024)

	ref_statevector = ref_result.get_statevector()
	mock_statevector = mock_result.get_statevector()  # type: ignore

	assert np.allclose(ref_statevector, mock_statevector)


def test_single_qubit_gates() -> None:
	"""Test a single qubit with a sequence of gates."""
	qc = QuantumCircuit(1)
	qc.x(0)  # Flip to |1>
	qc.h(0)  # Put into superposition

	qc.save_statevector()  # type: ignore

	ref_sim = reference_simulator()
	mock_sim = mock_simulator()

	ref_result = ref_sim.run(qc).result()
	mock_result = mock_sim.run(qc, shots=1024)

	ref_statevector = ref_result.get_statevector()
	mock_statevector = mock_result.get_statevector()  # type: ignore

	assert np.allclose(ref_statevector, mock_statevector)
