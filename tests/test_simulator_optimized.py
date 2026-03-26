"""Tests and benchmarks for optimized simulator parity and performance."""

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.random import random_circuit

from fp_qsim.simulator import CustomSimulatorManual
from fp_qsim.simulator_optimized import CustomSimulatorManualOptimized
from fp_qsim.state_vector import mocked_statevector


def align_global_phase(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Align candidate statevector to reference by removing global phase.

    Args:
        reference: Reference statevector.
        candidate: Candidate statevector to be phase-aligned.

    Returns:
        Phase-aligned candidate statevector.

    """
    anchor = int(np.argmax(np.abs(candidate)))
    if np.isclose(candidate[anchor], 0.0):
        return candidate
    return candidate * (reference[anchor] / candidate[anchor])


def make_cx_heavy_circuit(n_qubits: int, layers: int) -> QuantumCircuit:
    """Build a CX-dominant circuit to stress the manual CX kernel path.

    Args:
        n_qubits: Number of qubits in the circuit.
        layers: Number of alternating CX layers.

    Returns:
        Transpiled circuit in the ["u", "cx"] basis.

    """
    qc = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc.h(q)

    for _ in range(layers):
        for q in range(0, n_qubits - 1, 2):
            qc.cx(q, q + 1)
        for q in range(1, n_qubits - 1, 2):
            qc.cx(q, q + 1)

    return transpile(qc, basis_gates=["u", "cx"])


def test_optimized_matches_reference_bell() -> None:
    """Verify optimized simulator matches reference on a Bell circuit.

    Returns:
        None.

    """
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    circuit_ucx = transpile(qc, basis_gates=["u", "cx"])

    sim = CustomSimulatorManualOptimized(cx_backend="python")
    result = sim.run(circuit_ucx)
    reference = mocked_statevector(circuit_ucx)

    assert np.allclose(reference, result)


def test_optimized_numba_matches_reference_random() -> None:
    """Verify numba backend matches reference on a random transpiled circuit.

    Returns:
        None.

    """
    qc = random_circuit(4, 10, measure=False, seed=42)
    circuit_ucx = transpile(qc, basis_gates=["u", "cx"])

    sim = CustomSimulatorManualOptimized(cx_backend="numba")
    result = sim.run(circuit_ucx)
    reference = mocked_statevector(circuit_ucx)

    aligned = align_global_phase(reference, result)
    assert np.allclose(reference, aligned)


def test_python_and_numba_backends_match() -> None:
    """Verify python and numba backends produce equivalent statevectors.

    Returns:
        None.

    """
    qc = random_circuit(5, 14, measure=False, seed=123)
    circuit_ucx = transpile(qc, basis_gates=["u", "cx"])

    python_result = CustomSimulatorManualOptimized(cx_backend="python").run(circuit_ucx)
    numba_result = CustomSimulatorManualOptimized(cx_backend="numba").run(circuit_ucx)

    aligned = align_global_phase(python_result, numba_result)
    assert np.allclose(python_result, aligned)


def test_run_batch_equivalent_to_serial() -> None:
    """Verify run_batch results are equivalent to serial run outputs.

    Returns:
        None.

    """
    circuits = [
        transpile(random_circuit(4, 8, measure=False, seed=seed), basis_gates=["u", "cx"]) for seed in [11, 22, 33]
    ]

    sim = CustomSimulatorManualOptimized(cx_backend="python")
    serial = [sim.run(circuit) for circuit in circuits]
    batch = sim.run_batch(circuits, max_workers=2)

    assert len(batch) == len(serial)
    for expected, candidate in zip(serial, batch, strict=False):
        aligned = align_global_phase(expected, candidate)
        assert np.allclose(expected, aligned)


@pytest.mark.benchmark(group="cx-heavy-runtime")
@pytest.mark.parametrize("n_qubits", range(5, 17))
def test_benchmark_cx_heavy_baseline_manual(benchmark: BenchmarkFixture, n_qubits: int) -> None:
    """Benchmark CX-heavy runtime for the manual baseline simulator.

    Args:
        benchmark: pytest-benchmark fixture used to time execution.
        n_qubits: Number of qubits in the generated benchmark circuit.

    Returns:
        None.

    """
    circuit = make_cx_heavy_circuit(n_qubits=n_qubits, layers=4)
    simulator = CustomSimulatorManual()

    benchmark.extra_info["qubits"] = n_qubits
    benchmark.extra_info["simulator"] = "manual-baseline"
    benchmark(lambda: simulator.run(circuit, shots=1024))


@pytest.mark.benchmark(group="cx-heavy-runtime")
@pytest.mark.parametrize("n_qubits", range(5, 17))
def test_benchmark_cx_heavy_optimized(benchmark: BenchmarkFixture, n_qubits: int) -> None:
    """Benchmark CX-heavy runtime for the optimized simulator backend.

    Args:
        benchmark: pytest-benchmark fixture used to time execution.
        n_qubits: Number of qubits in the generated benchmark circuit.

    Returns:
        None.

    """
    circuit = make_cx_heavy_circuit(n_qubits=n_qubits, layers=4)
    simulator = CustomSimulatorManualOptimized(cx_backend="numba")

    benchmark.extra_info["qubits"] = n_qubits
    benchmark.extra_info["simulator"] = "manual-optimized"
    benchmark.extra_info["cx_backend"] = simulator.effective_cx_backend
    benchmark(lambda: simulator.run(circuit, shots=1024))
