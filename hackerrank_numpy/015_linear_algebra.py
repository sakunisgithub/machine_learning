import numpy as np

N = int(input())

A = np.empty((N, N), dtype = float)

for i in range(N) :
    temp = input().strip().split(' ')
    temp = [float(num) for num in temp]

    A[i] = temp

print(np.round(np.linalg.det(A), 2) )