import numpy as np

np.set_printoptions(legacy = '1.13')

dimensions = input().strip().split(' ')

dimensions = [int(num) for num in dimensions]

print(np.eye(dimensions[0], dimensions[1]))