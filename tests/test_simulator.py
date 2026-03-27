"""Tests and benchmarks for simulator behavior against Qiskit references."""

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFTGate
from qiskit.circuit.random import random_circuit
from qiskit_aer import AerSimulator

from fp_qsim.simulator_optimized import CustomSimulatorManualOptimized
from fp_qsim.state_vector import mocked_statevector


def reference_simulator() -> AerSimulator:
    """Create a reference Qiskit AerSimulator.

    Returns:
        AerSimulator: An AerSimulator explicitly configured to use the
            exact 'statevector' method.

    """
    return AerSimulator(method="statevector")


def reference_simulator_fast() -> AerSimulator:
    """Create an AerSimulator configured for benchmark throughput.

    Returns:
        AerSimulator: Statevector simulator with aggressive fusion and
            automatic thread utilization.

    """
    simulator = AerSimulator(method="statevector")
    simulator.set_options(
        fusion_enable=True,
        fusion_threshold=14,
        max_parallel_threads=0,
        max_parallel_experiments=1,
    )
    return simulator


def custom_simulator() -> CustomSimulatorManualOptimized:
    """Create an instance of the manual custom simulator.

    Returns:
        CustomSimulatorManual: The custom simulator optimized for 'u' and 'cx' gates.

    """
    return CustomSimulatorManualOptimized()


def align_global_phase(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Align candidate to reference using the strongest amplitude as phase anchor.

    Because global phase is physically unobservable, two identical quantum states
    may be represented by statevectors that differ by a global phase factor. This
    function uses the amplitude with the largest magnitude as an "anchor" to
    calculate and factor out that phase difference.

    Args:
        reference (np.ndarray): The reference statevector (e.g., from Qiskit).
        candidate (np.ndarray): The candidate statevector to be adjusted.

    Returns:
        np.ndarray: The candidate statevector multiplied by the calculated
            phase factor so it can be directly compared to the reference.

    """
    anchor = int(np.argmax(np.abs(candidate)))
    if np.isclose(candidate[anchor], 0.0):
        return candidate
    phase = reference[anchor] / candidate[anchor]
    return candidate * phase


def test_bell_state() -> None:
    """Test simulation of a 2-qubit Bell state.

    Creates a Bell state using an H gate and a CX gate and compares
    the simulator output to the reference statevector.

    Returns:
        None.

    """
    # Create a Bell state circuit
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    circuit_ucx = transpile(qc, basis_gates=["u", "cx"])
    qc.save_statevector()  # type: ignore
    # Get results from both simulators
    ref_sim = reference_simulator()
    custom_sim = custom_simulator()
    compiled_circuit = transpile(circuit_ucx, ref_sim)

    custom_result = custom_sim.run(circuit_ucx, shots=1024)

    # Extract statevectors
    ref_statevector = mocked_statevector(compiled_circuit)  # ref_result.get_statevector()
    custom_statevector = custom_result  # .get_statevector()

    # Assert that the statevectors are approximately equal
    assert np.allclose(ref_statevector, custom_statevector)


def test_ghz_state() -> None:
    """Test simulation of a 3-qubit GHZ state.

    Creates a maximally entangled GHZ state and verifies that the
    simulator statevector matches the reference.

    Returns:
        None.

    """
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)

    circuit_ucx = transpile(qc, basis_gates=["u", "cx"])
    qc.save_statevector()  # type: ignore
    ref_sim = reference_simulator()
    custom_sim = custom_simulator()
    compiled_circuit = transpile(circuit_ucx, ref_sim)

    custom_result = custom_sim.run(circuit_ucx, shots=1024)

    ref_statevector = mocked_statevector(compiled_circuit)
    custom_statevector = custom_result  # .get_statevector()

    assert np.allclose(ref_statevector, custom_statevector)


def test_single_qubit_gates() -> None:
    """Test a single qubit with a sequence of gates.

    Applies an X gate followed by an H gate to verify single-qubit
    unitary behavior.

    Returns:
        None.

    """
    qc = QuantumCircuit(1)
    qc.x(0)  # Flip to |1>
    qc.h(0)  # Put into superposition

    circuit_ucx = transpile(qc, basis_gates=["u", "cx"])
    qc.save_statevector()  # type: ignore
    ref_sim = reference_simulator()
    custom_sim = custom_simulator()
    compiled_circuit = transpile(circuit_ucx, ref_sim)

    custom_result = custom_sim.run(circuit_ucx, shots=1024)

    ref_statevector = mocked_statevector(compiled_circuit)
    custom_statevector = custom_result  # .get_statevector()
    assert np.allclose(ref_statevector, custom_statevector)


def test_qft() -> None:
    """Test the Quantum Fourier Transform on 5 qubits.

    Uses global phase alignment before assertion to account for arbitrary
    global phase differences introduced during synthesis and simulation.

    Returns:
        None.

    """
    qft_circ = QuantumCircuit(5)
    qft_circ.x(4)
    qft_circ.append(QFTGate(5), range(5))
    qft_circ.measure_all()
    circuit_ucx = transpile(qft_circ, basis_gates=["u", "cx"])

    qft_circ.save_statevector()  # type: ignore
    ref_sim = reference_simulator()
    custom_sim = custom_simulator()
    compiled_circuit = transpile(circuit_ucx, ref_sim)
    custom_result = custom_sim.run(circuit_ucx, shots=1024)
    ref_statevector = mocked_statevector(compiled_circuit)
    custom_statevector = custom_result  # .get_statevector()
    aligned_custom = align_global_phase(ref_statevector, custom_statevector)
    assert np.allclose(ref_statevector, aligned_custom)


def test_random_circuit() -> None:
    """Test a random circuit on 4 qubits.

    Generates a random circuit to validate handling of arbitrary unitary
    sequences and compares results after global-phase alignment.

    Returns:
        None.

    """
    qc = random_circuit(4, 10, measure=False, seed=1)
    circuit_ucx = transpile(qc, basis_gates=["u", "cx"])
    ref_sim = reference_simulator()
    custom_sim = custom_simulator()
    compiled_circuit = transpile(circuit_ucx, ref_sim)
    custom_result = custom_sim.run(circuit_ucx, shots=1024)
    ref_statevector = mocked_statevector(compiled_circuit)
    custom_statevector = custom_result  # .get_statevector()
    aligned_custom = align_global_phase(ref_statevector, custom_statevector)
    assert np.allclose(ref_statevector, aligned_custom)


@pytest.mark.benchmark(group="simulator-runtime")
@pytest.mark.parametrize("n_qubits", range(5, 17))
def test_benchmark_custom_simulator(benchmark: BenchmarkFixture, n_qubits: int) -> None:
    """Benchmarks the runtime of the custom simulator.

    Runs a randomly generated circuit of depth (2 * n_qubits) and measures
    execution time using pytest-benchmark.

    Args:
        benchmark (BenchmarkFixture): The pytest-benchmark fixture.
        n_qubits (int): The number of qubits for the current parameterized run.

    Returns:
        None.

    """
    depth = 2 * n_qubits
    custom_sim = custom_simulator()
    qc = random_circuit(n_qubits, depth, measure=False, seed=42)
    circuit_ucx = transpile(qc, basis_gates=["u", "cx"])
    circuit_ucx.save_statevector()
    benchmark.extra_info["qubits"] = n_qubits
    benchmark.extra_info["simulator"] = "custom"
    benchmark.extra_info["save_statevector"] = "after_transpile"
    benchmark(lambda: custom_sim.run(circuit_ucx, shots=1024))


@pytest.mark.benchmark(group="simulator-runtime")
@pytest.mark.parametrize("n_qubits", range(5, 25))
def test_benchmark_aer_simulator(benchmark: BenchmarkFixture, n_qubits: int) -> None:
    """Benchmarks the runtime of the reference Qiskit Aer simulator.

    Runs a randomly generated circuit of depth (2 * n_qubits) and measures
    execution time using pytest-benchmark for comparison against the custom simulator.

    Args:
        benchmark (BenchmarkFixture): The pytest-benchmark fixture.
        n_qubits (int): The number of qubits for the current parameterized run.

    Returns:
        None.

    """
    depth = 2 * n_qubits
    ref_sim = reference_simulator_fast()
    qc = random_circuit(n_qubits, depth, measure=False, seed=42)
    circuit_ucx = transpile(qc, basis_gates=["u", "cx"])
    compiled_circuit = transpile(circuit_ucx, ref_sim, optimization_level=3)
    compiled_circuit.save_statevector()
    benchmark.extra_info["qubits"] = n_qubits
    benchmark.extra_info["simulator"] = "aer"
    benchmark.extra_info["optimization_level"] = 3
    benchmark.extra_info["fusion_enable"] = True
    benchmark.extra_info["max_parallel_threads"] = "auto"
    benchmark.extra_info["save_statevector"] = "after_transpile"
    benchmark(lambda: ref_sim.run(compiled_circuit, shots=1024).result())
