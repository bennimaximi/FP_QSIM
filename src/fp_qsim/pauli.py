import numpy as np


def sigmax() -> np.ndarray:
	return np.array([[0, 1], [1, 0]], dtype=complex)


def sigmay() -> np.ndarray:
	return np.array([[0, -1j], [1j, 0]], dtype=complex)


def sigmaz() -> np.ndarray:
	return np.array([[1, 0], [0, -1]], dtype=complex)


sx: np.ndarray = sigmax()
sy: np.ndarray = sigmay()
sz: np.ndarray = sigmaz()

print('Sigma X:\n', sx)
print('Sigma Y:\n', sy)
print('Sigma Z:\n', sz)


# check equation:
left_side = sx @ sy
right_side = 1j * sz

if np.allclose(left_side, right_side):
	print('The equation sxsy = isz holds.')
else:
	print('The equation sxsy = isz does NOT hold.')
