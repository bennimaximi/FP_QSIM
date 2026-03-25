# pytest code for simulator.py
# import pytest
from qiskit.circuit.library import QFTGate
from qiskit import QuantumCircuit, transpile
from fp_qsim.simulator import CustomSimulatorGeneral
from qiskit_aer import AerSimulator
import numpy as np


def reference_simulator() -> AerSimulator:
	return AerSimulator(method='statevector')


def custom_simulator() -> CustomSimulatorGeneral:
	return CustomSimulatorGeneral()


def test_bell_state() -> None:
	# Create a Bell state circuit
	qc = QuantumCircuit(2)
	qc.h(0)
	qc.cx(0, 1)

	qc.save_statevector()  # type: ignore

	# Get results from both simulators
	ref_sim = reference_simulator()
	custom_sim = custom_simulator()

	ref_result = ref_sim.run(qc).result()
	custom_result = custom_sim.run(qc, shots=1024)

	# Extract statevectors
	ref_statevector = ref_result.get_statevector()
	custom_statevector = custom_result.get_statevector()  # type: ignore

	# Assert that the statevectors are approximately equal
	assert np.allclose(ref_statevector, custom_statevector)


def test_ghz_state() -> None:
	qc = QuantumCircuit(3)
	qc.h(0)
	qc.cx(0, 1)
	qc.cx(1, 2)

	qc.save_statevector()  # type: ignore

	ref_sim = reference_simulator()
	custom_sim = custom_simulator()

	ref_result = ref_sim.run(qc).result()
	custom_result = custom_sim.run(qc, shots=1024)

	ref_statevector = ref_result.get_statevector()
	custom_statevector = custom_result.get_statevector()  # type: ignore

	assert np.allclose(ref_statevector, custom_statevector)


def test_single_qubit_gates() -> None:
	"""Test a single qubit with a sequence of gates."""
	qc = QuantumCircuit(1)
	qc.x(0)  # Flip to |1>
	qc.h(0)  # Put into superposition

	qc.save_statevector()  # type: ignore

	ref_sim = reference_simulator()
	custom_sim = custom_simulator()

	ref_result = ref_sim.run(qc).result()
	custom_result = custom_sim.run(qc, shots=1024)

	ref_statevector = ref_result.get_statevector()
	custom_statevector = custom_result.get_statevector()  # type: ignore

	assert np.allclose(ref_statevector, custom_statevector)


def test_qft() -> None:
	"""Test the Quantum Fourier Transform on 3 qubits."""
	qft_circ = QuantumCircuit(5)
	qft_circ.x(4)
	qft_circ.append(QFTGate(5), range(5))
	qft_circ.save_statevector()  # type: ignore
	qft_circ.measure_all()
	ref_sim = reference_simulator()
	custom_sim = custom_simulator()
	compiled_circuit = transpile(qft_circ, ref_sim)
	ref_result = ref_sim.run(compiled_circuit).result()
	custom_result = custom_sim.run(compiled_circuit, shots=1024)
	ref_statevector = ref_result.get_statevector()
	custom_statevector = custom_result.get_statevector()  # type: ignore

	assert np.allclose(ref_statevector, custom_statevector)
