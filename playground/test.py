import numpy as np

from fp_qsim.pauli import sigmax, sigmay, sigmaz

# Test the Pauli matrices:

sx: np.ndarray = sigmax()
sy: np.ndarray = sigmay()
sz: np.ndarray = sigmaz()

print("Sigma X:\n", sx)
print("Sigma Y:\n", sy)
print("Sigma Z:\n", sz)


# check equation:
left_side = sx @ sy
right_side = 1j * sz

if np.allclose(left_side, right_side):
    print("The equation sxsy = isz holds.")
else:
    print("The equation sxsy = isz does NOT hold.")
