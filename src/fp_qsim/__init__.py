import numpy as np


def sigmax() -> np.ndarray:
	return np.array([[0, 1], [1, 0]], dtype=complex)


def sigmay() -> np.ndarray:
	return np.array([[0, -1j], [1j, 0]], dtype=complex)


def sigmaz() -> np.ndarray:
	return np.array([[1, 0], [0, -1]], dtype=complex)

def hello() -> str:
	return 'Hello from fp-qsim!'


def matrix_product(matrices: list[np.ndarray]) -> np.ndarray:
    """
    Multiply a list of matrices from left to right.

    Example:
        [A, B, C] -> A @ B @ C
    """
    if not matrices:
        raise ValueError("The input list must not be empty.")

    result = matrices[0]

    for matrix in matrices[1:]:
        if result.shape[1] != matrix.shape[0]:
            raise ValueError(
                f"Incompatible shapes for multiplication: "
                f"{result.shape} and {matrix.shape}"
            )
        result = result @ matrix

    return result
# from fp_qsim.pauli import sigmax, sigmay, sigmaz
