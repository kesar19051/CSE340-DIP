# Import the necessary libraries
from math import *
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

# input the path of the image
path = input("Enter the path of the image: ")

# load the image and convert into
# numpy array
img = Image.open(path)
img.show()
matrix = np.asarray(img)
# Plotting normalised histogram of the input image
gg = np.ndarray.flatten(np.array(matrix))
plt.hist(gg, bins = 256, density = True)
plt.show()
# matrix = matrix*(1.0)

# gamma transformatin
# c = 255/(max(np.amax(matrix))**(0.5))
gamma = 0.5
c = 255**(1-gamma)
# print(c)

gamma_matrix = c*(matrix**(0.5))

img_gamma = Image.fromarray(gamma_matrix)
img_gamma.show()

# Plotting normalised histogram of the input image
gg = np.ndarray.flatten(np.array(gamma_matrix))
plt.hist(gg, bins = 256, density = True)
plt.show()

nk_input = []
pk_input = {}
total = 65536

# Iterate for each pixel value
for px_val in range(256):
    indices = np.where(matrix==px_val)
    numOfIndices = np.size(indices)
    numOfIndices = numOfIndices/2
    nk_input.append(numOfIndices)
    pk_input[px_val] = numOfIndices/total

for i in range(2,256):
    pk_input[i] = pk_input[i]+pk_input[i-1]

nk_gamma = []
pk_gamma = {}
total = 65536

# Iterate for each pixel value
for px_val in range(256):
    indices = np.where(matrix==px_val)
    numOfIndices = np.size(indices)
    numOfIndices = numOfIndices/2
    nk_gamma.append(numOfIndices)
    pk_gamma[px_val] = numOfIndices/total

for i in range(1,256):
    pk_gamma[i] = pk_gamma[i]+pk_gamma[i-1]

mapped_values = {}

for px_val in range(256):
    H = pk_input[px_val]
    G = pk_gamma[0]
    mapped_values[px_val] = 0
    diff = abs(H-G)

    for i in range(1,256):
        G = pk_gamma[i]
        if(abs(H-G)<diff):
            diff = abs(H-G)
            mapped_values[px_val] = i

# print(mapped_values)

output_matrix = np.ones((256,256))*-1

for px_val in range(256):
    indices = np.where(matrix==px_val)
    size = np.size(indices)/2
    for i in range(int(size)):
        x = indices[0][i]
        y = indices[1][i]
        output_matrix[x][y] = mapped_values[px_val]

img_new = Image.fromarray(output_matrix)
img_new.show()

# Plotting normalised histogram of the output image
gg = np.ndarray.flatten(np.array(output_matrix))
plt.hist(gg, bins = 256, density = True)
plt.show()

# for i in range(256):
#     flag = False

#     for j in range(256):
#         a = matrix[i][j]
#         b = output_matrix[i][j]

#         a = int(a)
#         b = int(b)

#         if(a!=b):
#             print(False)
#             flag = True
#             break
    
#     if flag:
#         break