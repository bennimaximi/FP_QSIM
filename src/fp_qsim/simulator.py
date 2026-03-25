# mocked simulator for testing purposes
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from dataclasses import dataclass
import numpy as np
from qiskit.quantum_info import Operator


@dataclass
class MockSimulator(AerSimulator):
    """Wrapper around AerSimulator configured for statevector results."""

    def __init__(self) -> None:
        super().__init__()
        self._statevector_simulator = AerSimulator(method="statevector")

    def run(self, circuits: QuantumCircuit, shots: int) -> object:  # ty:ignore[invalid-method-override]
        """Execute the circuit and return the Qiskit result object.

        Args:
            circuits (QuantumCircuit): The quantum circuit to execute.
            shots (int): The number of shots for the simulation.

        Returns:
            object: Qiskit result object containing simulation data.
        """
        result = self._statevector_simulator.run(circuits, shots=shots).result()
        return result


@dataclass
class CustomSimulatorGeneral:
    """General-purpose statevector simulator based on tensor contraction.

    The simulator initializes an n-dimensional state tensor and applies each gate
    by contracting the state tensor with the corresponding unitary using
    ``numpy.einsum``.
    """

    def run(self, circuit: QuantumCircuit, shots: int = 1024) -> np.ndarray:
        """Simulate a circuit by applying gate tensors via ``numpy.einsum``.

        Args:
            circuit (QuantumCircuit): The quantum circuit to simulate.
            shots (int): Number of shots. Defaults to 1024. This value is currently
                unused because this simulator computes a deterministic statevector.

        Returns:
            np.ndarray: Flattened complex statevector.
        """
        n_qubits = circuit.num_qubits

        # Tensor axes are ordered as [q_{n-1}, ..., q_0] so flattening matches Qiskit's basis order.
        state = np.zeros([2] * n_qubits, dtype=complex)
        state[(0,) * n_qubits] = 1.0

        for instruction in circuit.data:
            gate = instruction.operation
            qubits = [circuit.find_bit(q).index for q in instruction.qubits][::-1]
            n_gate_qubits = len(qubits)
            if gate.name in [
                "measure",
                "barrier",
                "reset",
                "snapshot",
                "save_statevector",
                "load_statevector",
            ]:
                continue
            # Gate tensor has indices: [out_0, ..., out_k-1, in_0, ..., in_k-1].
            gate_tensor = Operator(gate).data.reshape([2] * (2 * n_gate_qubits))
            # print(Operator(gate).data)

            # Map logical qubits to tensor axes in state.
            state_axes = list(range(n_qubits))[::-1]
            qubit_axes = list(qubits)  # [n_qubits - 1 - q for q in qubits]

            out_axes = state_axes.copy()
            new_gate_out_axes = list(range(n_qubits, n_qubits + n_gate_qubits))
            for i, axis in enumerate(qubit_axes):
                out_axes[n_qubits - 1 - axis] = new_gate_out_axes[i]
            # print(new_gate_out_axes, qubit_axes, state_axes, out_axes)

            # Contract gate inputs with target state axes and keep updated state axes.
            state = np.einsum(
                gate_tensor,
                new_gate_out_axes + qubit_axes,
                state,
                state_axes,
                out_axes,
                optimize=False,
            )

        return state.reshape(-1)


@dataclass
class CustomSimulatorManual:
    """Manual simulator optimized for circuits in ``['u', 'cx']`` basis.

    It applies single-qubit unitaries with ``numpy.einsum`` and applies CX gates
    with direct amplitude swaps for performance and simplicity.
    """

    def apply_unitary(self, state: np.ndarray, gate_tensor: np.ndarray, qubits: list[int]) -> np.ndarray:
        """Apply a gate unitary to selected qubits on the state tensor.

        Args:
            state (np.ndarray): The current n-dimensional state tensor.
            gate_tensor (np.ndarray): The reshaped unitary matrix of the gate.
            qubits (list[int]): The indices of the qubits the gate acts upon.

        Returns:
            np.ndarray: Updated state tensor.
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
        """Apply a CX gate through direct amplitude swapping.

        This method avoids matrix multiplication, relying instead on bitwise operations
        to directly swap the relevant amplitudes in the flattened state array.

        Args:
            state (np.ndarray): The current n-dimensional state tensor.
            control (int): The index of the control qubit.
            target (int): The index of the target qubit.

        Returns:
            np.ndarray: Updated n-dimensional state tensor after CX.
        """
        n_qubits = state.ndim
        flat = state.reshape(-1).copy()
        n_states = 1 << n_qubits

        for i in range(n_states):
            if ((i >> control) & 1) == 1 and ((i >> target) & 1) == 0:
                j = i | (1 << target)
                flat[i], flat[j] = flat[j], flat[i]

        return flat.reshape([2] * n_qubits)

    def run(self, circuit: QuantumCircuit, shots: int = 1024) -> np.ndarray:
        """Simulate a circuit with manual ``u`` and ``cx`` gate handling.

        This method expects the circuit to ideally be composed of 'u' and 'cx' gates.
        Non-unitary instructions are ignored.

        Args:
            circuit (QuantumCircuit): The quantum circuit to simulate.
            shots (int, optional): Number of shots. Defaults to 1024.

        Returns:
            np.ndarray: Flattened complex statevector.
        """
        # circuit = transpile(circuit, basis_gates=['u', 'cx'])
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
            elif gate.name == "u":
                n_gate_qubits = len(qubits)
                # Qiskit's 'u' gate is a general single-qubit unitary, so we can directly use its matrix.
                gate_tensor = Operator(gate).data.reshape([2] * (2 * n_gate_qubits))
                state = self.apply_unitary(state, gate_tensor, qubits)
                continue
            elif gate.name == "cx":
                control, target = qubits
                state = self.apply_cx(state, control, target)

        return state.reshape(-1)
