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


sx = sigmax()
sy = sigmay()
sz = sigmaz()

print("Sigma X:\n", sx)
print("Sigma Y:\n", sy) 
print("Sigma Z:\n", sz)


#check equation:
left_side = sx @ sy
right_side = 1j * sz

if np.allclose(left_side, right_side):
    print("The equation σxσy = iσz holds.")
else:
    print("The equation σxσy = iσz does NOT hold.")
