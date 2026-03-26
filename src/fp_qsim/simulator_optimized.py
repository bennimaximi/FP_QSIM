from __future__ import annotations

import __main__
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing as mp
import sys
from typing import Literal

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator


try:
    import numba as nb  # type: ignore[import-not-found]

    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - import-time fallback
    nb = None  # type: ignore[assignment]
    _NUMBA_AVAILABLE = False


def _apply_cx_python_inplace(flat: np.ndarray, control: int, target: int) -> None:
    """Apply CX in-place by swapping amplitudes in a flat statevector."""
    n_states = flat.size
    target_mask = 1 << target

    for i in range(n_states):
        if ((i >> control) & 1) == 1 and ((i >> target) & 1) == 0:
            j = i | target_mask
            flat[i], flat[j] = flat[j], flat[i]


def _apply_u1_python_inplace(
    flat: np.ndarray,
    target: int,
    u00: complex,
    u01: complex,
    u10: complex,
    u11: complex,
) -> None:
    """Apply a single-qubit unitary in-place on a flat statevector."""
    n_states = flat.size
    target_mask = 1 << target

    for i in range(n_states):
        if (i & target_mask) == 0:
            j = i | target_mask
            a0 = flat[i]
            a1 = flat[j]
            flat[i] = u00 * a0 + u01 * a1
            flat[j] = u10 * a0 + u11 * a1


if _NUMBA_AVAILABLE:

    @nb.njit(cache=True, parallel=True)  # type: ignore[misc]
    def _apply_cx_numba_inplace(flat: np.ndarray, control: int, target: int) -> None:
        """Numba-parallel CX kernel for in-place amplitude swapping."""
        n_states = flat.size
        target_mask = 1 << target

        for i in nb.prange(n_states):
            if ((i >> control) & 1) == 1 and ((i >> target) & 1) == 0:
                j = i | target_mask
                temp = flat[i]
                flat[i] = flat[j]
                flat[j] = temp

    @nb.njit(cache=True, parallel=True)  # type: ignore[misc]
    def _apply_u1_numba_inplace(
        flat: np.ndarray,
        target: int,
        u00: complex,
        u01: complex,
        u10: complex,
        u11: complex,
    ) -> None:
        """Numba-parallel single-qubit unitary kernel."""
        n_states = flat.size
        target_mask = 1 << target

        for i in nb.prange(n_states):
            if (i & target_mask) == 0:
                j = i | target_mask
                a0 = flat[i]
                a1 = flat[j]
                flat[i] = u00 * a0 + u01 * a1
                flat[j] = u10 * a0 + u11 * a1

else:

    def _apply_cx_numba_inplace(flat: np.ndarray, control: int, target: int) -> None:
        """Fallback implementation if Numba is not available."""
        _apply_cx_python_inplace(flat, control, target)

    def _apply_u1_numba_inplace(
        flat: np.ndarray,
        target: int,
        u00: complex,
        u01: complex,
        u10: complex,
        u11: complex,
    ) -> None:
        """Fallback implementation if Numba is not available."""
        _apply_u1_python_inplace(flat, target, u00, u01, u10, u11)


@dataclass
class CustomSimulatorManualOptimized:
    """Manual simulator with optional multicore optimization.

    Supports:
    - Single-circuit speedups for CX gates using optional Numba parallelism.
    - Multi-circuit process-parallel execution via ``run_batch``.

    Args:
        cx_backend: Backend strategy for both ``u`` and ``cx`` kernels.
            - ``"auto"``: use Numba if available, otherwise Python fallback.
            - ``"numba"``: force Numba backend; raises if unavailable.
            - ``"python"``: force pure Python backend.
        num_threads: Optional Numba thread count for the CX kernel.
            Ignored for Python backend.
    """

    cx_backend: Literal["auto", "numba", "python"] = "auto"
    num_threads: int | None = None

    def __post_init__(self) -> None:
        if self.cx_backend == "numba" and not _NUMBA_AVAILABLE:
            raise ValueError("cx_backend='numba' requested, but numba is not installed.")

        if self.num_threads is not None and self.num_threads < 1:
            raise ValueError("num_threads must be >= 1 when provided.")

    def _configure_numba_threads(self) -> None:
        """Set Numba thread count when the Numba backend is active."""
        if self.effective_cx_backend == "numba" and self.num_threads is not None and _NUMBA_AVAILABLE:
            nb.set_num_threads(self.num_threads)

    @property
    def effective_cx_backend(self) -> Literal["numba", "python"]:
        """Return the backend that will actually be used."""
        if self.cx_backend == "python":
            return "python"
        if self.cx_backend == "numba":
            return "numba"
        return "numba" if _NUMBA_AVAILABLE else "python"

    def apply_unitary(self, state: np.ndarray, gate_tensor: np.ndarray, qubits: list[int]) -> np.ndarray:
        """Apply a gate unitary to selected qubits on the state tensor."""
        n_qubits = state.ndim
        state_axes = list(range(n_qubits))[::-1]
        qubit_axes = list(qubits)

        out_axes = state_axes.copy()
        new_gate_out_axes = list(range(n_qubits, n_qubits + len(qubits)))
        for i, axis in enumerate(qubit_axes):
            out_axes[n_qubits - 1 - axis] = new_gate_out_axes[i]

        return np.einsum(
            gate_tensor,
            new_gate_out_axes + qubit_axes,
            state,
            state_axes,
            out_axes,
            optimize=False,
        )

    def apply_cx(self, state: np.ndarray, control: int, target: int) -> np.ndarray:
        """Apply CX using the selected backend."""
        n_qubits = state.ndim
        flat = state.reshape(-1).copy()

        backend = self.effective_cx_backend
        if backend == "numba":
            _apply_cx_numba_inplace(flat, control, target)
        else:
            _apply_cx_python_inplace(flat, control, target)

        return flat.reshape([2] * n_qubits)

    def apply_u_single_qubit(self, state: np.ndarray, target: int, gate_matrix: np.ndarray) -> np.ndarray:
        """Apply a single-qubit unitary using backend-specific in-place kernels."""
        n_qubits = state.ndim
        flat = state.reshape(-1).copy()

        u00 = complex(gate_matrix[0, 0])
        u01 = complex(gate_matrix[0, 1])
        u10 = complex(gate_matrix[1, 0])
        u11 = complex(gate_matrix[1, 1])

        if self.effective_cx_backend == "numba":
            _apply_u1_numba_inplace(flat, target, u00, u01, u10, u11)
        else:
            _apply_u1_python_inplace(flat, target, u00, u01, u10, u11)

        return flat.reshape([2] * n_qubits)

    def run(self, circuit: QuantumCircuit, shots: int = 1024) -> np.ndarray:
        """Simulate a circuit with manual ``u`` and ``cx`` gate handling."""
        _ = shots
        n_qubits = circuit.num_qubits
        self._configure_numba_threads()

        # Tensor axes are ordered as [q_{n-1}, ..., q_0] so flattening matches Qiskit's basis order.
        state = np.zeros([2] * n_qubits, dtype=complex)
        state[(0,) * n_qubits] = 1.0

        for instruction in circuit.data:
            gate = instruction.operation
            qubits = [circuit.find_bit(q).index for q in instruction.qubits]

            if gate.name in [
                "measure",
                "barrier",
                "reset",
                "snapshot",
                "save_statevector",
                "load_statevector",
            ]:
                continue

            if gate.name == "u":
                target = qubits[0]
                gate_matrix = Operator(gate).data
                state = self.apply_u_single_qubit(state, target, gate_matrix)
                continue

            if gate.name == "cx":
                control, target = qubits
                state = self.apply_cx(state, control, target)
                continue

            # Fallback for other unitary gates to keep broad compatibility.
            n_gate_qubits = len(qubits)
            gate_tensor = Operator(gate).data.reshape([2] * (2 * n_gate_qubits))
            state = self.apply_unitary(state, gate_tensor, qubits)

        return state.reshape(-1)

    def run_batch(
        self,
        circuits: list[QuantumCircuit],
        shots: int = 1024,
        max_workers: int | None = None,
    ) -> list[np.ndarray]:
        """Run multiple circuits in parallel using process workers.

        For small batches, this method falls back to serial execution to avoid
        unnecessary process startup overhead.
        """
        if not circuits:
            return []

        if len(circuits) == 1:
            return [self.run(circuits[0], shots=shots)]

        worker_payload = [
            (circuit, shots, self.cx_backend, self.num_threads)
            for circuit in circuits
        ]

        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=_get_multiprocessing_context(),
        ) as executor:
            return list(executor.map(_run_single_circuit_worker, worker_payload))


def _get_multiprocessing_context() -> mp.context.BaseContext | None:
    """Return a multiprocessing context suitable for the current platform."""
    if sys.platform.startswith("win"):
        return None

    # In normal script/test execution, default context is safer.
    has_main_file = bool(getattr(__main__, "__file__", None))
    if has_main_file:
        return None

    # In interactive/STDIN sessions, spawn can fail because there is no __main__ file.
    return mp.get_context("fork")


def _run_single_circuit_worker(payload: tuple[QuantumCircuit, int, str, int | None]) -> np.ndarray:
    """Process worker entrypoint for run_batch."""
    circuit, shots, cx_backend, num_threads = payload
    sim = CustomSimulatorManualOptimized(
        cx_backend=cx_backend,  # type: ignore[arg-type]
        num_threads=num_threads,
    )
    return sim.run(circuit, shots=shots)
