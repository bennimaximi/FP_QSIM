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
    """Creates a reference Qiskit AerSimulator.

    Returns:
        AerSimulator: An AerSimulator explicitly configured to use the
            exact 'statevector' method.
    """
    return AerSimulator(method="statevector")


def custom_simulator() -> CustomSimulatorManual:
    """Creates an instance of the manual custom simulator.

    Returns:
        CustomSimulatorManual: The custom simulator optimized for 'u' and 'cx' gates.
    """
    return CustomSimulatorManual()


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
    """Tests the simulation of a 2-qubit Bell state.

    Creates a Bell state using an H gate and a CX gate,
    and asserts that the custom simulator output matches the reference.
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

    # ref_result = ref_sim.run(compiled_circuit).result()
    custom_result = custom_sim.run(circuit_ucx, shots=1024)

    # Extract statevectors
    ref_statevector = mocked_statevector(compiled_circuit)  # ref_result.get_statevector()
    custom_statevector = custom_result  # .get_statevector()

    # Assert that the statevectors are approximately equal
    assert np.allclose(ref_statevector, custom_statevector)


def test_ghz_state() -> None:
    """Tests the simulation of a 3-qubit GHZ state.

    Creates a 3-qubit maximally entangled GHZ state and asserts
    that the custom simulator's statevector matches the reference.
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

    # ref_result = ref_sim.run(compiled_circuit).result()
    custom_result = custom_sim.run(circuit_ucx, shots=1024)

    ref_statevector = mocked_statevector(compiled_circuit)
    custom_statevector = custom_result  # .get_statevector()

    assert np.allclose(ref_statevector, custom_statevector)


def test_single_qubit_gates() -> None:
    """Test a single qubit with a sequence of gates.
        Applies an X gate (flip to |1>) followed by an H gate
    to verify single-qubit unitary logic.
    """
    qc = QuantumCircuit(1)
    qc.x(0)  # Flip to |1>
    qc.h(0)  # Put into superposition

    circuit_ucx = transpile(qc, basis_gates=["u", "cx"])
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
    """Test the Quantum Fourier Transform on 3 qubits.
        Uses global phase alignment before the assertion to account for arbitrary phase differences introduced
    during QFT synthesis and simulation.
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
    # ref_result = ref_sim.run(compiled_circuit).result()
    custom_result = custom_sim.run(circuit_ucx, shots=1024)
    ref_statevector = mocked_statevector(compiled_circuit)
    custom_statevector = custom_result  # .get_statevector()
    aligned_custom = align_global_phase(ref_statevector, custom_statevector)
    assert np.allclose(ref_statevector, aligned_custom)


def test_random_circuit() -> None:
    """Test a random circuit on 4 qubits.

        Generates a random 4-qubit circuit to ensure the
    custom simulator can handle an arbitrary sequence of unitary gates.
    Phase alignment is used before comparison.
    """
    qc = random_circuit(4, 10, measure=False, seed=1)
    # qc.save_statevector()
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
@pytest.mark.parametrize("n_qubits", range(5, 13))
def test_benchmark_custom_simulator(benchmark: Any, n_qubits: int) -> None:
    """Benchmarks the runtime of the custom simulator.

    Runs a randomly generated circuit of depth (2 * n_qubits) and measures
    execution time using pytest-benchmark.

    Args:
        benchmark (Any): The pytest-benchmark fixture.
        n_qubits (int): The number of qubits for the current parameterized run.
    """
    depth = 2 * n_qubits
    custom_sim = custom_simulator()
    qc = random_circuit(n_qubits, depth, measure=False, seed=42)
    circuit_ucx = transpile(qc, basis_gates=["u", "cx"])

    benchmark.extra_info["qubits"] = n_qubits
    benchmark.extra_info["simulator"] = "custom"
    benchmark(lambda: custom_sim.run(circuit_ucx, shots=1024))


@pytest.mark.benchmark(group="simulator-runtime")
@pytest.mark.parametrize("n_qubits", range(5, 13))
def test_benchmark_aer_simulator(benchmark: Any, n_qubits: int) -> None:
    """Benchmarks the runtime of the reference Qiskit Aer simulator.

    Runs a randomly generated circuit of depth (2 * n_qubits) and measures
    execution time using pytest-benchmark for comparison against the custom simulator.

    Args:
        benchmark (Any): The pytest-benchmark fixture.
        n_qubits (int): The number of qubits for the current parameterized run.
    """
    depth = 2 * n_qubits
    ref_sim = reference_simulator()
    qc = random_circuit(n_qubits, depth, measure=False, seed=42)
    circuit_ucx = transpile(qc, basis_gates=["u", "cx"])
    compiled_circuit = transpile(circuit_ucx, ref_sim)

    benchmark.extra_info["qubits"] = n_qubits
    benchmark.extra_info["simulator"] = "aer"
    benchmark(lambda: ref_sim.run(compiled_circuit, shots=1024).result())
