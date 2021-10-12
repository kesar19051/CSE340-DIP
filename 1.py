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

# the padded matrix
padded_matrix = np.ones((2*M,2*N))*0

# padding the image to 2M X 2N
for i in range(M):
    for j in range(N):
        padded_matrix[i][j] = matrix[i][j]

# display padded image
img = Image.fromarray(padded_matrix)
img.show()

# the centered 2d dft of image
centered_2d_dft = np.ones((2*M,2*N))*0

# computing the centered dft by multiplying with (-1)^(n+m)
for i in range(2*M):
    for j in range(2*N):
        centered_2d_dft[i][j] = padded_matrix[i][j]*pow(-1,i+j)

img = Image.fromarray(centered_2d_dft)
img.show()

# finding the dft
dft_image_matrix = np.fft.fft2(centered_2d_dft)

# for plotting the magnitude spectrum
magnitude_spectrum = np.ones((2*M,2*N))*0

# filling the values of magnitude spectrum
for i in range(2*M):
    for j in range(2*N):
        magnitude_spectrum[i][j] = int(round(abs(dft_image_matrix[i][j])))

# finding the min and max value in the matrix for scaling
max = np.amax(magnitude_spectrum)
min = np.amin(magnitude_spectrum)

# scaling the pixel values
for i in range(2*M):
    for j in range(2*N):
        magnitude_spectrum[i][j] = ((magnitude_spectrum[i][j]-min)/(max-min))*255

# display the magnitude spectrum
img = Image.fromarray(magnitude_spectrum)
img.show()

# defining the filter
filter = np.ones((2*M,2*N))*0

# taking in D0
D0 = int(input("Enter the value of D0: "))
n = int(input("Enter the value of n: "))

# defining D(u,v)
def Duv(i,j):
    i = pow(i-256,2)
    j = pow(j-256,2)
    return pow(i+j,0.5)

# creating the filter
for i in range(2*M):
    for j in range(2*N):
        filter[i][j] = 1/(1+pow((Duv(i,j)/D0),2*n))

filter_to_show = np.ones((2*M,2*N))*0

max = np.amax(filter)
min = np.amin(filter)

for i in range(2*M):
    for j in range(2*N):
        filter_to_show[i][j] = ((filter[i][j]-min)/(max-min))*255

img = Image.fromarray(filter_to_show)
img.show()

# taking elementwise multiplication
elementwise_multiplied = np.multiply(dft_image_matrix,filter)

# computing the inverse dft
idft = np.fft.ifftn(elementwise_multiplied)

# taking the real part of the matrix
real_idft = idft.real

# centering it
for i in range(2*M):
    for j in range(2*N):
        real_idft[i][j] = real_idft[i][j]*pow(-1,i+j)

img = Image.fromarray(real_idft)
img.show()

# cropping the final image
output_image = np.ones((M,N))*0

for i in range(M):
    for j in range(N):
        output_image[i][j] = real_idft[i][j]

img = Image.fromarray(output_image)
img.show()