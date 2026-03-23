import numpy as np

def sigmax():
    return np.array([[0, 1], 
                     [1, 0]] , dtype=complex)

def sigmay():
    return np.array([[0, -1j], 
                     [1j, 0]], dtype=complex)

def sigmaz():
    return np.array([[1, 0], 
                     [0, -1]], dtype=complex)


# 1. Retrieve the matrices
sx = sigmax()
sy = sigmay()
sz = sigmaz()

print("Sigma X:\n", sx)
print("Sigma Y:\n", sy) 
print("Sigma Z:\n", sz)