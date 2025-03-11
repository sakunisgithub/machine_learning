numbers = input().strip().split(' ')

numbers = [int(num) for num in numbers]

import numpy as np

A = np.empty((numbers[0], numbers[1]), dtype = np.int64)

for i in range(numbers[0]) :
    temp = input().strip().split(' ')
    temp = [int(num) for num in temp]

    A[i] = temp

print(np.prod(np.sum(A, axis = 0)))