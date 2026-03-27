"""Tests and benchmarks for CUDA simulator parity and performance."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.random import random_circuit

from fp_qsim.simulator_gpu import CustomSimulatorManualGPU
from fp_qsim.simulator_optimized import CustomSimulatorManualOptimized
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


def measure_runtime_seconds(
    run_callable: Callable[[], np.ndarray],
    *,
    repeats: int = 3,
    warmup_runs: int = 1,
) -> np.ndarray:
    """Measure repeated runtime for a simulator run callable.

    Args:
        run_callable: Callable that executes one simulation run.
        repeats: Number of measured runs.
        warmup_runs: Number of unmeasured warmup runs.

    Returns:
        Array of runtimes in seconds.

    """
    for _ in range(warmup_runs):
        run_callable()

    timings: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        run_callable()
        timings.append(time.perf_counter() - start)

    return np.asarray(timings, dtype=np.float64)


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
def test_cuda_matches_numba_random_20q_depth10() -> None:
    """Verify CUDA and Numba backends produce equivalent states.

    Returns:
        None.

    """
    qc = random_circuit(20, 10, measure=False, seed=2026)
    circuit_ucx = transpile(qc, basis_gates=["u", "cx"])

    gpu_result = CustomSimulatorManualGPU().run(circuit_ucx)
    numba_result = CustomSimulatorManualOptimized(cx_backend="numba").run(circuit_ucx)

    aligned_gpu = align_global_phase(numba_result, gpu_result)
    assert np.allclose(numba_result, aligned_gpu)


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
def test_gpu_advantage_vs_numba_random_20q_depth10() -> None:
    """Compare GPU and Numba runtimes on a fixed random 20q depth-10 circuit.

    Returns:
        None.

    """
    qc = random_circuit(20, 10, measure=False, seed=2027)
    circuit_ucx = transpile(qc, basis_gates=["u", "cx"])

    gpu_sim = CustomSimulatorManualGPU()
    numba_sim = CustomSimulatorManualOptimized(cx_backend="numba")

    # Trigger JIT/kernel warmup once before timed runs.
    _ = numba_sim.run(circuit_ucx)
    _ = gpu_sim.run(circuit_ucx)

    gpu_times = measure_runtime_seconds(lambda: gpu_sim.run(circuit_ucx), repeats=3, warmup_runs=1)
    numba_times = measure_runtime_seconds(lambda: numba_sim.run(circuit_ucx), repeats=3, warmup_runs=1)

    gpu_median = float(np.median(gpu_times))
    numba_median = float(np.median(numba_times))
    speedup = numba_median / gpu_median

    print("\n=== GPU vs Numba Runtime Comparison (20 qubits, depth 10) ===")
    print(f"GPU runs (ms):   {[round(1000.0 * t, 2) for t in gpu_times]}")
    print(f"Numba runs (ms): {[round(1000.0 * t, 2) for t in numba_times]}")
    print(f"GPU median (ms):   {1000.0 * gpu_median:.2f}")
    print(f"Numba median (ms): {1000.0 * numba_median:.2f}")
    print(f"Speedup (numba/gpu): {speedup:.2f}x")
    print("============================================================\n")

    assert speedup > 1.0, f"Expected GPU advantage over numba, got speedup={speedup:.2f}x"


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
