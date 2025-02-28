sizes = input().strip().split(' ')

sizes = [int(num) for num in sizes]

N = sizes[0]; M = sizes[1]; P = sizes[2]

import numpy as np

arr1 = np.empty((N, P), dtype = np.int32)

for i in range(N):
    temp = input().strip().split(' ') 

    temp = [int(num) for num in temp]

    arr1[i] = temp

arr2 = np.empty((M, P), dtype = np.int32)

for i in range(M):
    temp = input().strip().split(' ')

    temp = [int(num) for num in temp]

    arr2[i] = temp

print(np.concatenate((arr1, arr2), axis = 0))