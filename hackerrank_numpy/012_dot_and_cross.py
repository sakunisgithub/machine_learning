N = int(input())

import numpy as np

A = np.empty((N, N), dtype = np.int64)
B = np.empty((N, N), dtype = np.int64)

for i in range(N) :
    temp = input().strip().split(' ')
    temp = [int(num) for num in temp]

    A[i] = temp

for i in range(N) :
    temp = input().strip().split(' ')
    temp = [int(num) for num in temp]

    B[i] = temp

print(np.dot(A, B))