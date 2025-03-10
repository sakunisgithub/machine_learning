A = input().strip().split(' ')

import numpy as np

np.set_printoptions(legacy = '1.13')

A = np.array([float(num) for num in A])

print(np.floor(A))

print(np.ceil(A))

print(np.rint(A))