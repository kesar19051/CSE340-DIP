from math import *
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

C = 3
R = 3

def reverseColumns(arr):
    for i in range(C):
        j = 0
        k = C-1
        while j < k:
            t = arr[j][i]
            arr[j][i] = arr[k][i]
            arr[k][i] = t
            j += 1
            k -= 1
             
# Function for transpose of matrix
def transpose(arr):
    for i in range(R):
        for j in range(i, C):
            t = arr[i][j]
            arr[i][j] = arr[j][i]
            arr[j][i] = t

filter = np.array([[1,9,3],[0,5,6],[7,8,9]])

print(filter)
transpose(filter)
reverseColumns(filter)
transpose(filter)
reverseColumns(filter)
print(filter)