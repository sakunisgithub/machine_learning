sizes = input().strip().split(' ') 

sizes = tuple([int(num) for num in sizes])

import numpy as np

arr1 = np.zeros(sizes, dtype = np.int8)

arr2 = np.ones(sizes, dtype = np.int8)

print(arr1)

print(arr2)