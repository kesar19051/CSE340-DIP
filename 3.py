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
img.show()
matrix = np.asarray(img)

# M = number of rows; N = number of columns
M = len(matrix)
N = len(matrix[0])

# the box filter
convolution_matrix = np.ones((9,9))*(1/81)

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

# the elementwise multiplication
elementwise_multiplied = np.ones((M+9-1,N+9-1))*-1
elementwise_multiplied = np.multiply(dft_convolution_matrix,dft_padded_matrix)

# computing the inverse dft
idft = np.fft.ifftn(elementwise_multiplied)

# taking the real part of the matrix
real_idft = idft.real

# the output matrix
new_matrix = np.ones((M,N))*-1

# Transferring the value to the new matrix
for i in range(M):
    for j in range(N):
        new_matrix[i][j] = round(real_idft[i][j])

# the output image
img_new = Image.fromarray(new_matrix)
img_new.show()

# covolution done using in-built function and output shown
output_matrix = signal.convolve2d(matrix,convolution_matrix)
img_new_ = Image.fromarray(output_matrix)
img_new_.show()