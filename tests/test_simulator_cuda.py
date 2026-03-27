"""Tests and benchmarks for CUDA simulator parity and performance."""

from __future__ import annotations

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.random import random_circuit

from fp_qsim.simulator_gpu import CustomSimulatorManualGPU
from fp_qsim.state_vector import mocked_statevector

try:
    from numba import cuda
except Exception:
    HAS_CUDA = False
else:
    HAS_CUDA = bool(cuda.is_available())


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
    """Build a CX-dominant circuit to stress the CUDA CX kernel path.

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


def test_cuda_backend_availability_contract() -> None:
    """Ensure simulator fails fast without CUDA and initializes with CUDA.

    Returns:
        None.

    """
    if HAS_CUDA:
        simulator = CustomSimulatorManualGPU()
        assert simulator.effective_cx_backend == "cuda"
        return

    with pytest.raises(RuntimeError, match="CUDA"):
        CustomSimulatorManualGPU()


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
def test_cuda_matches_reference_bell() -> None:
    """Verify CUDA simulator matches reference on a Bell circuit.

    Returns:
        None.

    """
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    circuit_ucx = transpile(qc, basis_gates=["u", "cx"])

    sim = CustomSimulatorManualGPU()
    result = sim.run(circuit_ucx)
    reference = mocked_statevector(circuit_ucx)

    aligned = align_global_phase(reference, result)
    assert np.allclose(reference, aligned)


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
def test_cuda_matches_reference_random() -> None:
    """Verify CUDA simulator matches reference on a random transpiled circuit.

    Returns:
        None.

    """
    qc = random_circuit(6, 16, measure=False, seed=2026)
    circuit_ucx = transpile(qc, basis_gates=["u", "cx"])

    sim = CustomSimulatorManualGPU()
    result = sim.run(circuit_ucx)
    reference = mocked_statevector(circuit_ucx)

    aligned = align_global_phase(reference, result)
    assert np.allclose(reference, aligned)


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
@pytest.mark.benchmark(group="cuda-runtime")
@pytest.mark.parametrize("n_qubits", range(8, 25))
def test_benchmark_cuda_runtime(benchmark: BenchmarkFixture, n_qubits: int) -> None:
    """Benchmark runtime of the CUDA simulator over random circuits.

    Args:
        benchmark: pytest-benchmark fixture used to time execution.
        n_qubits: Number of qubits in the generated benchmark circuit.

    Returns:
        None.

    """
    depth = 2 * n_qubits
    circuit = random_circuit(n_qubits, depth, measure=False, seed=42)
    circuit_ucx = transpile(circuit, basis_gates=["u", "cx"])

    simulator = CustomSimulatorManualGPU()
    benchmark.extra_info["qubits"] = n_qubits
    benchmark.extra_info["simulator"] = "manual-gpu"
    benchmark.extra_info["cx_backend"] = simulator.effective_cx_backend
    benchmark.extra_info["threads_per_block"] = simulator.threads_per_block

    benchmark(lambda: simulator.run(circuit_ucx, shots=1024))


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
@pytest.mark.benchmark(group="cx-heavy-runtime-cuda")
@pytest.mark.parametrize("n_qubits", range(8, 25))
def test_benchmark_cuda_cx_heavy(benchmark: BenchmarkFixture, n_qubits: int) -> None:
    """Benchmark CUDA simulator on CX-heavy circuits.

    Args:
        benchmark: pytest-benchmark fixture used to time execution.
        n_qubits: Number of qubits in the generated benchmark circuit.

    Returns:
        None.

    """
    circuit = make_cx_heavy_circuit(n_qubits=n_qubits, layers=4)
    simulator = CustomSimulatorManualGPU()

    benchmark.extra_info["qubits"] = n_qubits
    benchmark.extra_info["simulator"] = "manual-gpu"
    benchmark.extra_info["cx_backend"] = simulator.effective_cx_backend
    benchmark.extra_info["threads_per_block"] = simulator.threads_per_block

    benchmark(lambda: simulator.run(circuit, shots=1024))
