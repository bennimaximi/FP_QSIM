"""CUDA-backed simulator kernels and backend selection logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

try:
    from numba import cuda
except Exception as exc:  # pragma: no cover - exercised only when numba.cuda import fails.
    cuda = None
    _CUDA_IMPORT_ERROR = exc
else:
    _CUDA_IMPORT_ERROR = None


if cuda is not None:

    @cuda.jit
    def _apply_u1_cuda_kernel(
        flat_in: np.ndarray,
        flat_out: np.ndarray,
        target_stride: int,
        pair_stride: int,
        lower: int,
        n_pairs: int,
        u00: complex,
        u01: complex,
        u10: complex,
        u11: complex,
    ) -> None:
        """Apply one single-qubit gate pass over independent amplitude pairs."""
        pair_idx = cuda.grid(1)
        if pair_idx >= n_pairs:
            return

        upper_idx = pair_idx // lower
        lower_idx = pair_idx - (upper_idx * lower)
        upper_base = upper_idx * pair_stride

        i0 = upper_base + lower_idx
        i1 = i0 + target_stride

        a0 = flat_in[i0]
        a1 = flat_in[i1]

        flat_out[i0] = u00 * a0 + u01 * a1
        flat_out[i1] = u10 * a0 + u11 * a1


    @cuda.jit
    def _apply_cx_cuda_kernel(flat: np.ndarray, control: int, target: int, n_states: int) -> None:
        """Apply a CX gate in place using one-sided swap ownership."""
        idx = cuda.grid(1)
        if idx >= n_states:
            return

        control_on = (idx >> control) & 1
        target_on = (idx >> target) & 1
        if control_on == 1 and target_on == 0:
            partner = idx | (1 << target)
            temp = flat[idx]
            flat[idx] = flat[partner]
            flat[partner] = temp

else:

    def _apply_u1_cuda_kernel(*args: object, **kwargs: object) -> None:
        """Raise when CUDA kernel is unavailable in this environment."""
        msg = "numba.cuda import failed; CUDA kernels are unavailable."
        raise RuntimeError(msg)


    def _apply_cx_cuda_kernel(*args: object, **kwargs: object) -> None:
        """Raise when CUDA kernel is unavailable in this environment."""
        msg = "numba.cuda import failed; CUDA kernels are unavailable."
        raise RuntimeError(msg)


@dataclass
class CustomSimulatorManualGPU:
    """Manual simulator using Numba CUDA kernels for ``u`` and ``cx`` gates.

    Args:
        threads_per_block: CUDA threads per block. Must be a positive multiple of 32.
        min_active_blocks: Preferred lower bound for launched blocks to saturate SMs.

    """

    threads_per_block: int = 256
    min_active_blocks: int = 144
    _device_sm_count: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate CUDA availability and kernel launch configuration.

        Raises:
            RuntimeError: If CUDA support is unavailable.
            ValueError: If launch configuration values are invalid.

        """
        if self.threads_per_block <= 0 or self.threads_per_block % 32 != 0:
            raise ValueError("threads_per_block must be a positive multiple of 32.")
        if self.min_active_blocks <= 0:
            raise ValueError("min_active_blocks must be positive.")

        if cuda is None:
            msg = "numba.cuda import failed; install CUDA-enabled numba to use CustomSimulatorManualGPU."
            if _CUDA_IMPORT_ERROR is not None:
                raise RuntimeError(msg) from _CUDA_IMPORT_ERROR
            raise RuntimeError(msg)

        if not cuda.is_available():
            raise RuntimeError("CUDA is not available. This simulator is CUDA-only.")

        device = cuda.get_current_device()
        self._device_sm_count = int(device.MULTIPROCESSOR_COUNT)

    @property
    def effective_cx_backend(self) -> Literal["cuda"]:
        """Return backend identifier for benchmark metadata.

        Returns:
            Literal["cuda"]: Constant backend identifier.

        """
        return "cuda"

    def _grid_for_work(self, work_items: int) -> int:
        """Compute launch block count with occupancy-aware rounding.

        Args:
            work_items: Number of parallel work items.

        Returns:
            Number of blocks to launch.

        """
        if work_items <= 0:
            return 1

        required_blocks = (work_items + self.threads_per_block - 1) // self.threads_per_block
        sm_target = max(self.min_active_blocks, self._device_sm_count)

        if required_blocks <= sm_target:
            return sm_target

        return ((required_blocks + sm_target - 1) // sm_target) * sm_target

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

    def _apply_u_device(
        self,
        d_flat_in: np.ndarray,
        target: int,
        u00: complex,
        u01: complex,
        u10: complex,
        u11: complex,
    ) -> np.ndarray:
        """Launch the CUDA u-kernel and return output device buffer.

        Args:
            d_flat_in: Input device statevector.
            target: Target qubit index.
            u00: Matrix element (0, 0).
            u01: Matrix element (0, 1).
            u10: Matrix element (1, 0).
            u11: Matrix element (1, 1).

        Returns:
            Output device statevector buffer.

        """
        n_states = int(d_flat_in.size)
        lower = 2**target
        target_stride = lower
        pair_stride = 2 ** (target + 1)
        n_pairs = n_states // 2

        d_flat_out = cuda.device_array_like(d_flat_in)
        blocks = self._grid_for_work(n_pairs)

        _apply_u1_cuda_kernel[blocks, self.threads_per_block](
            d_flat_in,
            d_flat_out,
            target_stride,
            pair_stride,
            lower,
            n_pairs,
            u00,
            u01,
            u10,
            u11,
        )
        return d_flat_out

    def _apply_cx_device_inplace(self, d_flat: np.ndarray, control: int, target: int) -> None:
        """Launch the CUDA CX kernel in place.

        Args:
            d_flat: Device statevector (flattened).
            control: Control qubit index.
            target: Target qubit index.

        Returns:
            None.

        """
        if control == target:
            return

        n_states = int(d_flat.size)
        blocks = self._grid_for_work(n_states)
        _apply_cx_cuda_kernel[blocks, self.threads_per_block](d_flat, control, target, n_states)

    def apply_cx(self, state: np.ndarray, control: int, target: int) -> np.ndarray:
        """Apply a CX gate with CUDA and return updated host tensor.

        Args:
            state: Current n-dimensional state tensor.
            control: Control qubit index.
            target: Target qubit index.

        Returns:
            Updated state tensor.

        """
        n_qubits = state.ndim
        flat = state.reshape(-1).copy()
        d_flat = cuda.to_device(flat)
        self._apply_cx_device_inplace(d_flat, control, target)
        return d_flat.copy_to_host().reshape([2] * n_qubits)

    def apply_u_single_qubit(self, state: np.ndarray, target: int, gate_matrix: np.ndarray) -> np.ndarray:
        """Apply a single-qubit unitary with CUDA and return updated host tensor.

        Args:
            state: Current n-dimensional state tensor.
            target: Target qubit index.
            gate_matrix: 2x2 unitary matrix.

        Returns:
            Updated state tensor.

        """
        n_qubits = state.ndim
        flat = state.reshape(-1).copy()
        d_flat = cuda.to_device(flat)

        u00 = complex(gate_matrix[0, 0])
        u01 = complex(gate_matrix[0, 1])
        u10 = complex(gate_matrix[1, 0])
        u11 = complex(gate_matrix[1, 1])

        d_flat_out = self._apply_u_device(d_flat, target, u00, u01, u10, u11)
        return d_flat_out.copy_to_host().reshape([2] * n_qubits)

    def run(self, circuit: QuantumCircuit, shots: int = 1024) -> np.ndarray:
        """Simulate a circuit with CUDA-accelerated ``u`` and ``cx`` gate paths.

        Args:
            circuit: Quantum circuit to simulate.
            shots: Shot count placeholder (not used for deterministic statevector output).

        Returns:
            Flattened complex statevector.

        """
        _ = shots
        n_qubits = circuit.num_qubits

        flat = np.zeros(2**n_qubits, dtype=np.complex128)
        flat[0] = 1.0 + 0.0j
        d_flat = cuda.to_device(flat)
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

                d_flat = self._apply_u_device(d_flat, target, u00, u01, u10, u11)
                continue

            if gate.name == "cx":
                control, target = qubits
                self._apply_cx_device_inplace(d_flat, control, target)
                continue

            gate_tensor = Operator(gate).data.reshape([2] * (2 * len(qubits)))
            state_tensor = d_flat.copy_to_host().reshape(tensor_shape)
            state_tensor = self.apply_unitary(state_tensor, gate_tensor, qubits)
            d_flat = cuda.to_device(state_tensor.reshape(-1))

        cuda.synchronize()
        return d_flat.copy_to_host()

    def run_batch(
        self,
        circuits: list[QuantumCircuit],
        shots: int = 1024,
        max_workers: int | None = None,
    ) -> list[np.ndarray]:
        """Run multiple circuits serially with the CUDA backend.

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
    

