dimensions = input().strip().split(' ')

dimensions = [int(num) for num in dimensions]

N = dimensions[0]; M = dimensions[1]

import numpy as np

A = np.empty((N, M), dtype = np.int64)

B = np.empty((N, M), dtype = np.int64)

for i in range(N):
    temp = input().strip().split(' ')
    temp = [int(num) for num in temp]

    A[i] = temp

for i in range(N):
    temp = input().strip().split(' ')
    temp = [int(num) for num in temp]

    B[i] = temp

print(np.add(A, B))

print(np.subtract(A, B))

print(np.multiply(A, B))

print(A // B)

print(np.mod(A, B))

print(np.power(A, B))