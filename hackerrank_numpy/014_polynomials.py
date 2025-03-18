import numpy as np

coefficients = input().strip().split(' ')

coefficients = [float(num) for num in coefficients]

x = float(input())

print(np.polyval(coefficients, x))