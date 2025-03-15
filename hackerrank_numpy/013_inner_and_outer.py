import numpy as np

A = input().strip().split(' ') 

A = [int(num) for num in A]

A = np.array(A)

B = input().strip().split(' ') 

B = [int(num) for num in B]

B = np.array(B)

print(np.inner(A, B))

print(np.outer(A, B))