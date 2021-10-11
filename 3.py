# Import the necessary libraries
from math import *
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

# input the path of the image
path = 'camera.jpg'

# load the image and convert into
# numpy array
img = Image.open(path)
matrix = np.asarray(img)

# M = number of rows; N = number of columns
M = len(matrix)
N = len(matrix[0])

convolution_matrix = np.ones((9,9))*(1/81)

new_matrix = np.ones((M+9-1,N+9-1))*-1

# padded image matrix
padded_matrix = np.ones((M+9-1,N+9-1))*0

# padded convolution matrix
convolution_padded_matrix = np.ones((M+9-1,N+9-1))*0

# padding with 0s the image matrix
for i in range(M):
    for j in range(N):
        padded_matrix[i][j] = matrix[i][j]

# padding with 0s the covolution matrix
for i in range(9):
    for j in range(9):
        convolution_padded_matrix[i][j] = convolution_matrix[i][j]

# finding the dft of both matrices
dft_padded_matrix = np.fft.fft2(padded_matrix)
dft_convolution_matrix = np.fft.fft2(convolution_padded_matrix)

print(dft_padded_matrix)
print(dft_convolution_matrix)

elementwise_multiplied = np.ones((M+9-1,N+9-1))*-1

# multiplying elementwise
# for i in range(M+9-1):
#     for j in range(N+9-1):
#         elementwise_multiplied[i][j] = dft_convolution_matrix[i][j]*dft_padded_matrix[i][j]

elementwise_multiplied = np.multiply(dft_convolution_matrix,dft_padded_matrix)

print(elementwise_multiplied)

# computing the inverse dft
idft = np.fft.ifftn(elementwise_multiplied)

print("element", elementwise_multiplied)

# taking the real part of the matrix
real_idft = idft.real

new_matrix = np.ones((M,N))*-1

for i in range(M):
    for j in range(N):
        new_matrix[i][j] = round(real_idft[i][j])

# output_matrix = np.ones((M,N))*-1
# for i in range(M):
#     for j in range(N):
#         output_matrix[i][j] = new_matrix[i][j]

img_new = Image.fromarray(new_matrix)

img_new.show()

output_matrix = signal.convolve2d(matrix,convolution_matrix)

img_new_ = Image.fromarray(output_matrix)

img_new_.show()
