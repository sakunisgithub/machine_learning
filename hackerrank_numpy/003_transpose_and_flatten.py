dimensions = input().strip().split(' ')

dimensions = [int(dim) for dim in dimensions]

import numpy as np

numbers = np.empty((dimensions[0], dimensions[1]), dtype = np.int32)

for i in range(dimensions[0]) :
    temp = input().strip().split(' ')

    temp = [int(num) for num in temp]

    numbers[i] = temp

print(numbers.T)

print(numbers.flatten())