from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numba as nb
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator


def _apply_cx_python_inplace(flat: np.ndarray, control: int, target: int) -> None:
    """Apply CX using explicit lower/upper index traversal around target."""
    n_states = flat.size
    lower_block = 2 ** target
    pair_block = 2 ** (target + 1)
    control_mask = 2 ** control

    for upper_base in range(0, n_states, pair_block):
        for lower in range(lower_block):
            i0 = upper_base + lower
            i1 = i0 + lower_block
            if (i0 & control_mask) != 0:
                temp = flat[i0]
                flat[i0] = flat[i1]
                flat[i1] = temp


def _apply_u1_python_inplace(
    flat: np.ndarray,
    target: int,
    u00: complex,
    u01: complex,
    u10: complex,
    u11: complex,
) -> None:
    """Apply 2x2 unitary on one qubit via explicit index-pair loops.

    Basis indexing is little-endian: flat index = sum(bit_q * 2**q).
    """
    n_states = flat.size
    lower_block = 2 ** target
    pair_block = 2 ** (target + 1)

    for upper_base in range(0, n_states, pair_block):
        for lower in range(lower_block):
            i0 = upper_base + lower
            i1 = i0 + lower_block
            a0 = flat[i0]
            a1 = flat[i1]
            flat[i0] = u00 * a0 + u01 * a1
            flat[i1] = u10 * a0 + u11 * a1


@nb.njit(cache=True)  # type: ignore[misc]
def _apply_cx_numba_inplace(flat: np.ndarray, control: int, target: int) -> None:
    """Numba CX kernel with the same explicit lower/upper index traversal."""
    n_states = flat.size
    lower_block = 2 ** target
    pair_block = 2 ** (target + 1)
    control_mask = 2 ** control

    for upper_base in range(0, n_states, pair_block):
        for lower in range(lower_block):
            i0 = upper_base + lower
            i1 = i0 + lower_block
            if (i0 & control_mask) != 0:
                temp = flat[i0]
                flat[i0] = flat[i1]
                flat[i1] = temp


@nb.njit(cache=True)  # type: ignore[misc]
def _apply_u1_numba_inplace(
    flat: np.ndarray,
    target: int,
    u00: complex,
    u01: complex,
    u10: complex,
    u11: complex,
) -> None:
    """Numba single-qubit kernel with explicit target-bit pair traversal."""
    n_states = flat.size
    lower_block = 2 ** target
    pair_block = 2 ** (target + 1)

    for upper_base in range(0, n_states, pair_block):
        for lower in range(lower_block):
            i0 = upper_base + lower
            i1 = i0 + lower_block
            a0 = flat[i0]
            a1 = flat[i1]
            flat[i0] = u00 * a0 + u01 * a1
            flat[i1] = u10 * a0 + u11 * a1


@dataclass
class CustomSimulatorManualOptimized:
    """Manual simulator with explicit loop kernels for ``u`` and ``cx``.

    Supports:
    - Pure Python loop kernels for transparent indexing logic.
    - Numba JIT kernels (cache enabled) with the same loop structure.
    - Serial batch execution via ``run_batch``.

    Args:
        cx_backend: Explicit backend choice for ``u`` and ``cx`` kernels.
            - ``"python"``: use pure Python loops.
            - ``"numba"``: use Numba JIT-compiled loops.
    """

    cx_backend: Literal["numba", "python"] = "python"

    def __post_init__(self) -> None:
        if self.cx_backend not in {"python", "numba"}:
            raise ValueError("cx_backend must be either 'python' or 'numba'.")

    @property
    def effective_cx_backend(self) -> Literal["numba", "python"]:
        """Return the explicitly selected backend."""
        return self.cx_backend

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
        """Run multiple circuits serially with the selected backend."""
        _ = max_workers
        if not circuits:
            return []

        return [self.run(circuit, shots=shots) for circuit in circuits]
