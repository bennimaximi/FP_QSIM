import numpy as np


def sigmax() -> np.ndarray:
	return np.array([[0, 1], [1, 0]], dtype=complex)


def sigmay() -> np.ndarray:
	return np.array([[0, -1j], [1j, 0]], dtype=complex)


def sigmaz() -> np.ndarray:
	return np.array([[1, 0], [0, -1]], dtype=complex)


