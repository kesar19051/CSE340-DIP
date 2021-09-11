# Import the necessary libraries
from math import *
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

# Defining the image and filter
array = np.random.randint(10, size=(3, 3))
# array = np.array([[0,0,0],[0,1,0],[0,0,0]])
filter = np.random.randint(10, size=(3,3))
# filter = np.array([[1,2,3],[4,5,6],[7,8,9]])

# Initialising rotated filter
rotatedFilter = np.ones((3,3))*-1

# Padding the image matrix
padded_matrix = np.ones((5,5))*0

for x in range(1,4):
    for y in range(1,4):
        padded_matrix[x][y] = array[x-1][y-1]

print()

# Printing the image matrix
print("The image matrix: ")
print(array)
print()

# Printing the filter
print("The filter: ")
print(filter)
print()

# Rotating the filter
rotatedFilter[0][0] = filter[2][2]
rotatedFilter[0][1] = filter[2][1]
rotatedFilter[0][2] = filter[2][0]
rotatedFilter[1][0] = filter[1][2]
rotatedFilter[1][1] = filter[1][1]
rotatedFilter[1][2] = filter[1][0]
rotatedFilter[2][0] = filter[0][2]
rotatedFilter[2][1] = filter[0][1]
rotatedFilter[2][2] = filter[0][0]

# Printing the rotated filter
print("The rotated filter: ")
print(rotatedFilter)
print()

# Initialising the output matrix
output_matrix = np.ones((5,5))*-1

# Convoluting the image
for i in range(5):
    for j in range(5):
        list = []

        if i-1<5 and i-1>=0 and j-1<5 and j-1>=0:
            a1 = padded_matrix[i-1][j-1]
            list.append(a1)
        else:
            list.append(0)
        if i-1<5 and i-1>=0 and j<5 and j>=0:
            a2 = padded_matrix[i-1][j]
            list.append(a2)
        else:
            list.append(0)
        if i-1<5 and i-1>=0 and j+1<5 and j+1>=0:
            a3 = padded_matrix[i-1][j+1]
            list.append(a3)
        else:
            list.append(0)
        if j-1<5 and j-1>=0 and i>=0 and i<5:
            a4 = padded_matrix[i][j-1]
            list.append(a4)
        else:
            list.append(0)
        if i<5 and i>=0 and j<5 and j>=0:
            a5 = padded_matrix[i][j]
            list.append(a5)
        else:
            list.append(0)
        if j+1<5 and j+1>=0 and i<5 and i>=0:
            a6 = padded_matrix[i][j+1]
            list.append(a6)
        else:
            list.append(0)
        if i+1<5 and i+1>=0 and j-1<5 and j-1>=0:
            a7 = padded_matrix[i+1][j-1]
            list.append(a7)
        else:
            list.append(0)
        if i+1<5 and i+1>=0 and j>=0 and j<5:
            a8 = padded_matrix[i+1][j]
            list.append(a8)
        else:
            list.append(0)
        if i+1<5 and i+1>=0 and j+1<5 and j+1>=0:
            a9 = padded_matrix[i+1][j+1]
            list.append(a9)
        else:
            list.append(0)

        index = 0
        pixel_value = 0

        for x in range(3):
            for y in range(3):
                pixel_value = pixel_value+rotatedFilter[x][y]*list[index]
                index = index+1
        
        output_matrix[i][j] = pixel_value

print("The output matrix: ")
print(output_matrix)