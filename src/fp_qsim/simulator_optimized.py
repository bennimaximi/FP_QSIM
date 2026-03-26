"""Optimized custom simulator kernels and backend selection logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numba as nb
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator


def _apply_cx_python_inplace(flat: np.ndarray, control: int, target: int) -> None:
    """Apply a CX gate in place using explicit lower/upper index traversal.

    Args:
        flat: Flattened statevector in little-endian basis order.
        control: Control qubit index.
        target: Target qubit index.

    Returns:
        None. The input array is mutated in place.

    """
    if control == target:
        return

    n_states = flat.size
    lower_block = 2**target
    pair_block = 2 ** (target + 1)

    if control < target:
        control_block = 2**control
        lower_stride = 2 ** (control + 1)

        for upper_base in range(0, n_states, pair_block):
            for lower_start in range(control_block, lower_block, lower_stride):
                lower_end = lower_start + control_block
                for lower in range(lower_start, lower_end):
                    i0 = upper_base + lower
                    i1 = i0 + lower_block
                    temp = flat[i0]
                    flat[i0] = flat[i1]
                    flat[i1] = temp
        return

    upper_control_bit = 2 ** (control - target - 1)
    upper_count = n_states // pair_block
    for upper in range(upper_count):
        if (upper & upper_control_bit) == 0:
            continue

        upper_base = upper * pair_block
        for lower in range(lower_block):
            i0 = upper_base + lower
            i1 = i0 + lower_block
            temp = flat[i0]
            flat[i0] = flat[i1]
            flat[i1] = temp


def _apply_u1_loop_body(
    flat_in: np.ndarray,
    flat_out: np.ndarray,
    lower: int,
    upper: int,
    target: int,
    u00: complex,
    u01: complex,
    u10: complex,
    u11: complex,
) -> None:
    """Apply one single-qubit update pass using shared upper/lower traversal.

    Args:
        flat_in: Input flattened statevector.
        flat_out: Output flattened statevector.
        lower: Number of lower indices, equal to 2**target.
        upper: Number of upper index blocks.
        target: Target qubit index.
        u00: Matrix element (0, 0) of the 2x2 gate.
        u01: Matrix element (0, 1) of the 2x2 gate.
        u10: Matrix element (1, 0) of the 2x2 gate.
        u11: Matrix element (1, 1) of the 2x2 gate.

    Returns:
        None. Results are written to flat_out.

    """
    target_stride = 2**target
    pair_stride = 2 ** (target + 1)

    for upper_idx in range(upper):
        upper_base = upper_idx * pair_stride
        for lower_idx in range(lower):
            i0 = lower_idx + upper_base
            i1 = i0 + target_stride
            a0 = flat_in[i0]
            a1 = flat_in[i1]
            flat_out[i0] = u00 * a0 + u01 * a1
            flat_out[i1] = u10 * a0 + u11 * a1


def _apply_u1_python(
    flat: np.ndarray,
    target: int,
    u00: complex,
    u01: complex,
    u10: complex,
    u11: complex,
) -> np.ndarray:
    """Apply a 2x2 unitary to one qubit using Python loop traversal.

    Args:
        flat: Input flattened statevector.
        target: Target qubit index.
        u00: Matrix element (0, 0) of the 2x2 gate.
        u01: Matrix element (0, 1) of the 2x2 gate.
        u10: Matrix element (1, 0) of the 2x2 gate.
        u11: Matrix element (1, 1) of the 2x2 gate.

    Returns:
        Updated flattened statevector.

    """
    lower = 2**target
    upper = flat.size // (2 ** (target + 1))
    flat_out = np.empty_like(flat)
    _apply_u1_loop_body(flat, flat_out, lower, upper, target, u00, u01, u10, u11)
    return flat_out


@nb.njit(cache=True)  # type: ignore[misc]
def _apply_cx_numba_inplace(flat: np.ndarray, control: int, target: int) -> None:
    """Apply a CX gate in place using the Numba-compiled CX traversal.

    Args:
        flat: Flattened statevector in little-endian basis order.
        control: Control qubit index.
        target: Target qubit index.

    Returns:
        None. The input array is mutated in place.

    """
    if control == target:
        return

    n_states = flat.size
    lower_block = 2**target
    pair_block = 2 ** (target + 1)

    if control < target:
        control_block = 2**control
        lower_stride = 2 ** (control + 1)

        for upper_base in range(0, n_states, pair_block):
            for lower_start in range(control_block, lower_block, lower_stride):
                lower_end = lower_start + control_block
                for lower in range(lower_start, lower_end):
                    i0 = upper_base + lower
                    i1 = i0 + lower_block
                    temp = flat[i0]
                    flat[i0] = flat[i1]
                    flat[i1] = temp
        return

    upper_control_bit = 2 ** (control - target - 1)
    upper_count = n_states // pair_block
    for upper in range(upper_count):
        if (upper & upper_control_bit) == 0:
            continue

        upper_base = upper * pair_block
        for lower in range(lower_block):
            i0 = upper_base + lower
            i1 = i0 + lower_block
            temp = flat[i0]
            flat[i0] = flat[i1]
            flat[i1] = temp


_apply_u1_numba_kernel = nb.njit(cache=True)(_apply_u1_loop_body)


def _apply_u1_numba(
    flat: np.ndarray,
    target: int,
    u00: complex,
    u01: complex,
    u10: complex,
    u11: complex,
) -> np.ndarray:
    """Apply a 2x2 unitary to one qubit using a Numba-compiled loop kernel.

    Args:
        flat: Input flattened statevector.
        target: Target qubit index.
        u00: Matrix element (0, 0) of the 2x2 gate.
        u01: Matrix element (0, 1) of the 2x2 gate.
        u10: Matrix element (1, 0) of the 2x2 gate.
        u11: Matrix element (1, 1) of the 2x2 gate.

    Returns:
        Updated flattened statevector.

    """
    lower = 2**target
    upper = flat.size // (2 ** (target + 1))
    flat_out = np.empty_like(flat)
    _apply_u1_numba_kernel(flat, flat_out, lower, upper, target, u00, u01, u10, u11)
    return flat_out


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
        """Validate backend selection after dataclass initialization.

        Raises:
            ValueError: If cx_backend is not one of "python" or "numba".

        """
        if self.cx_backend not in {"python", "numba"}:
            raise ValueError("cx_backend must be either 'python' or 'numba'.")

    @property
    def effective_cx_backend(self) -> Literal["numba", "python"]:
        """Return the explicitly selected backend.

        Returns:
            Backend identifier, either "numba" or "python".

        """
        return self.cx_backend

    def apply_unitary(self, state: np.ndarray, gate_tensor: np.ndarray, qubits: list[int]) -> np.ndarray:
        """Apply a gate tensor to selected qubits of the state tensor.

        Args:
            state: Current n-dimensional state tensor.
            gate_tensor: Gate tensor reshaped as [out..., in...].
            qubits: Logical qubit indices acted on by the gate.

        Returns:
            Updated state tensor.

        """
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
        """Apply a CX gate using the selected backend.

        Args:
            state: Current n-dimensional state tensor.
            control: Control qubit index.
            target: Target qubit index.

        Returns:
            Updated state tensor after applying CX.

        """
        n_qubits = state.ndim
        flat = state.reshape(-1).copy()

        backend = self.effective_cx_backend
        if backend == "numba":
            _apply_cx_numba_inplace(flat, control, target)
        else:
            _apply_cx_python_inplace(flat, control, target)

        return flat.reshape([2] * n_qubits)

    def apply_u_single_qubit(self, state: np.ndarray, target: int, gate_matrix: np.ndarray) -> np.ndarray:
        """Apply a single-qubit unitary using backend-specific loop kernels.

        Args:
            state: Current n-dimensional state tensor.
            target: Target qubit index.
            gate_matrix: 2x2 unitary matrix.

        Returns:
            Updated state tensor after applying the single-qubit gate.

        """
        n_qubits = state.ndim
        flat = state.reshape(-1).copy()

        u00 = complex(gate_matrix[0, 0])
        u01 = complex(gate_matrix[0, 1])
        u10 = complex(gate_matrix[1, 0])
        u11 = complex(gate_matrix[1, 1])

        if self.effective_cx_backend == "numba":
            flat = _apply_u1_numba(flat, target, u00, u01, u10, u11)
        else:
            flat = _apply_u1_python(flat, target, u00, u01, u10, u11)

        return flat.reshape([2] * n_qubits)

    def run(self, circuit: QuantumCircuit, shots: int = 1024) -> np.ndarray:
        """Simulate a circuit with manual u and cx gate handling.

        Args:
            circuit: Quantum circuit to simulate.
            shots: Shot count placeholder (not used for deterministic statevector output).

        Returns:
            Flattened complex statevector.

        """
        _ = shots
        n_qubits = circuit.num_qubits

        # Keep a flat statevector throughout the loop to avoid repeated reshape/copy work.
        flat = np.zeros(2**n_qubits, dtype=complex)
        flat[0] = 1.0
        tensor_shape = [2] * n_qubits

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
                theta, phi, lam = gate.params
                theta_f = float(theta)
                phi_f = float(phi)
                lam_f = float(lam)

                cos_half = np.cos(theta_f / 2.0)
                sin_half = np.sin(theta_f / 2.0)
                phase_phi = np.exp(1j * phi_f)
                phase_lam = np.exp(1j * lam_f)
                phase_phi_lam = phase_phi * phase_lam

                u00 = complex(cos_half)
                u01 = complex(-phase_lam * sin_half)
                u10 = complex(phase_phi * sin_half)
                u11 = complex(phase_phi_lam * cos_half)

                if self.effective_cx_backend == "numba":
                    flat = _apply_u1_numba(flat, target, u00, u01, u10, u11)
                else:
                    flat = _apply_u1_python(flat, target, u00, u01, u10, u11)
                continue

            if gate.name == "cx":
                control, target = qubits
                if self.effective_cx_backend == "numba":
                    _apply_cx_numba_inplace(flat, control, target)
                else:
                    _apply_cx_python_inplace(flat, control, target)
                continue

            # Fallback for other unitary gates to keep broad compatibility.
            n_gate_qubits = len(qubits)
            gate_tensor = Operator(gate).data.reshape([2] * (2 * n_gate_qubits))
            state_tensor = flat.reshape(tensor_shape)
            state_tensor = self.apply_unitary(state_tensor, gate_tensor, qubits)
            flat = state_tensor.reshape(-1)

        return flat

    def run_batch(
        self,
        circuits: list[QuantumCircuit],
        shots: int = 1024,
        max_workers: int | None = None,
    ) -> list[np.ndarray]:
        """Run multiple circuits serially with the selected backend.

        Args:
            circuits: Circuits to execute.
            shots: Shot count placeholder forwarded to run.
            max_workers: Unused compatibility parameter.

        Returns:
            List of flattened statevectors, one per circuit.

        """
        _ = max_workers
        if not circuits:
            return []

        return [self.run(circuit, shots=shots) for circuit in circuits]
