numbers = input().strip().split(' ')

numbers = [int(num) for num in numbers]

N = numbers[0]; M = numbers[1]

import numpy as np

A = np.empty((N, M), dtype = np.int64)

for i in range(N) :
    temp = input().strip().split(' ')
    temp = [int(num) for num in temp]

    A[i] = temp


print(np.max(np.min(A, axis = 1)))