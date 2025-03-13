numbers = input().strip().split(' ')

numbers = [int(num) for num in numbers]

N = numbers[0]; M = numbers[1]

import numpy as np

A = np.empty((N, M), dtype = np.int64)

for i in range(N) :
    temp = input().strip().split(' ')
    temp = [int(num) for num in temp]

    A[i] = temp

print(np.mean(A, axis = 1))

print(np.var(A, axis = 0))

print(np.round(np.std(A, axis = None), 11))
# rounding off to 11 digits was necessary for passing hackerrank test cases