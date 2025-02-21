import numpy

def arrays(arr):
    temp = numpy.array(arr, dtype = float)
    return(temp[::-1])

arr = input().strip().split(' ')
result = arrays(arr)
print(result)