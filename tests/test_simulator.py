# pytest code for simulator.py
import pytest
# import glob
from typing import Any
from qiskit.circuit.library import QFTGate
from qiskit import QuantumCircuit, transpile
from fp_qsim.simulator import CustomSimulatorManual
from qiskit_aer import AerSimulator
import numpy as np
from qiskit.circuit.random import random_circuit
from fp_qsim.state_vector import mocked_statevector


def reference_simulator() -> AerSimulator:
	return AerSimulator(method='statevector')


def custom_simulator() -> CustomSimulatorManual:
	return CustomSimulatorManual()


def test_bell_state() -> None:
	# Create a Bell state circuit
	qc = QuantumCircuit(2)
	qc.h(0)
	qc.cx(0, 1)

	circuit_ucx = transpile(qc, basis_gates=['u', 'cx'])
	qc.save_statevector()  # type: ignore
	# Get results from both simulators
	ref_sim = reference_simulator()
	custom_sim = custom_simulator()
	compiled_circuit = transpile(circuit_ucx, ref_sim)

	# ref_result = ref_sim.run(compiled_circuit).result()
	custom_result = custom_sim.run(circuit_ucx, shots=1024)

	# Extract statevectors
	ref_statevector = mocked_statevector(compiled_circuit)  # ref_result.get_statevector()
	custom_statevector = custom_result  # .get_statevector()

	# Assert that the statevectors are approximately equal
	assert np.allclose(ref_statevector, custom_statevector)


def test_ghz_state() -> None:
	qc = QuantumCircuit(3)
	qc.h(0)
	qc.cx(0, 1)
	qc.cx(1, 2)

	circuit_ucx = transpile(qc, basis_gates=['u', 'cx'])
	qc.save_statevector()  # type: ignore
	ref_sim = reference_simulator()
	custom_sim = custom_simulator()
	compiled_circuit = transpile(circuit_ucx, ref_sim)

	# ref_result = ref_sim.run(compiled_circuit).result()
	custom_result = custom_sim.run(circuit_ucx, shots=1024)

	ref_statevector = mocked_statevector(compiled_circuit)
	custom_statevector = custom_result  # .get_statevector()

	assert np.allclose(ref_statevector, custom_statevector)


def test_single_qubit_gates() -> None:
	"""Test a single qubit with a sequence of gates."""
	qc = QuantumCircuit(1)
	qc.x(0)  # Flip to |1>
	qc.h(0)  # Put into superposition

	circuit_ucx = transpile(qc, basis_gates=['u', 'cx'])
	qc.save_statevector()  # type: ignore
	ref_sim = reference_simulator()
	custom_sim = custom_simulator()
	compiled_circuit = transpile(circuit_ucx, ref_sim)

	# ref_result = ref_sim.run(compiled_circuit).result()
	custom_result = custom_sim.run(circuit_ucx, shots=1024)

	ref_statevector = mocked_statevector(compiled_circuit)
	custom_statevector = custom_result  # .get_statevector()
	assert np.allclose(ref_statevector, custom_statevector)


def test_qft() -> None:
	"""Test the Quantum Fourier Transform on 3 qubits."""
	qft_circ = QuantumCircuit(5)
	qft_circ.x(4)
	qft_circ.append(QFTGate(5), range(5))
	qft_circ.measure_all()
	circuit_ucx = transpile(qft_circ, basis_gates=['u', 'cx'])

	qft_circ.save_statevector()  # type: ignore
	ref_sim = reference_simulator()
	custom_sim = custom_simulator()
	compiled_circuit = transpile(circuit_ucx, ref_sim)
	# ref_result = ref_sim.run(compiled_circuit).result()
	custom_result = custom_sim.run(circuit_ucx, shots=1024)
	ref_statevector = mocked_statevector(compiled_circuit)
	custom_statevector = custom_result  # .get_statevector()
	global_phase = ref_statevector[0] / custom_statevector[0]
	assert np.allclose(ref_statevector, custom_statevector * global_phase)


def test_random_circuit() -> None:
	"""Test a random circuit on 4 qubits."""
	qc = random_circuit(4, 10, measure=False)
	# qc.save_statevector()  # type: ignore
	circuit_ucx = transpile(qc, basis_gates=['u', 'cx'])
	ref_sim = reference_simulator()
	custom_sim = custom_simulator()
	compiled_circuit = transpile(circuit_ucx, ref_sim)
	custom_result = custom_sim.run(circuit_ucx, shots=1024)
	ref_statevector = mocked_statevector(compiled_circuit)
	custom_statevector = custom_result  # .get_statevector()
	global_phase = ref_statevector[0] / custom_statevector[0]
	assert np.allclose(ref_statevector, custom_statevector * global_phase)


@pytest.mark.benchmark(group='simulator-runtime')
@pytest.mark.parametrize('n_qubits', range(5, 13))
def test_benchmark_custom_simulator(benchmark: Any, n_qubits: int) -> None:
	"""Benchmark custom simulator runtime from 5 to 12 qubits."""
	depth = 2 * n_qubits
	custom_sim = custom_simulator()
	qc = random_circuit(n_qubits, depth, measure=False, seed=42)
	circuit_ucx = transpile(qc, basis_gates=['u', 'cx'])

	benchmark.extra_info['qubits'] = n_qubits
	benchmark.extra_info['simulator'] = 'custom'
	benchmark(lambda: custom_sim.run(circuit_ucx, shots=1024))


@pytest.mark.benchmark(group='simulator-runtime')
@pytest.mark.parametrize('n_qubits', range(5, 13))
def test_benchmark_aer_simulator(benchmark: Any, n_qubits: int) -> None:
	"""Benchmark Aer simulator runtime from 5 to 12 qubits."""
	depth = 2 * n_qubits
	ref_sim = reference_simulator()
	qc = random_circuit(n_qubits, depth, measure=False, seed=42)
	circuit_ucx = transpile(qc, basis_gates=['u', 'cx'])
	compiled_circuit = transpile(circuit_ucx, ref_sim)

	benchmark.extra_info['qubits'] = n_qubits
	benchmark.extra_info['simulator'] = 'aer'
	benchmark(lambda: ref_sim.run(compiled_circuit, shots=1024).result())
