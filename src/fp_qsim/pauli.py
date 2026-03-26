"""Pauli matrix helper functions."""

import numpy as np


def sigmax() -> np.ndarray:
    """Return the Pauli-X matrix.

    Returns:
        Complex 2x2 array for the X operator.

    """
    return np.array([[0, 1], [1, 0]], dtype=complex)


def sigmay() -> np.ndarray:
    """Return the Pauli-Y matrix.

    Returns:
        Complex 2x2 array for the Y operator.

    """
    return np.array([[0, -1j], [1j, 0]], dtype=complex)


def sigmaz() -> np.ndarray:
    """Return the Pauli-Z matrix.

    Returns:
        Complex 2x2 array for the Z operator.

    """
    return np.array([[1, 0], [0, -1]], dtype=complex)
