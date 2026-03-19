import scipy.linalg as la 
import numpy as np
A = np.array([[1., 1000], [0, 1]])
B = np.array([[1, 1000], [0.001, 1]])

print(A)

print(B)
wA, vrA = la.eig(A)
wB, vrB = la.eig(B)

print(wA, wB)
