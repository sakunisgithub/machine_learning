numbers = input().strip().split(' ')

numbers = [int(num) for num in numbers]

import numpy as np

print(np.array(numbers).reshape(3, 3))