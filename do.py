# Import the necessary libraries
from math import *
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

# input the path of the image
path = 'hello.jpg'

# load the image and convert into
# numpy array
img = Image.open(path)
img.show()
matrix = np.asarray(img)

# M = number of rows; N = number of columns
M = len(matrix)
N = len(matrix[0])

padded_matrix = np.ones((2*M,2*N))*0

# padding the image to 2M X 2N
for i in range(M):
    for j in range(N):
        padded_matrix[i][j] = matrix[i][j]

# dft of the image
dft_matrix = np.fft.fft2(padded_matrix)

# to display the magnitude spectrum
magnitude_spectrum = np.ones((M,N))*1

# displaying the magnitude spectrum
for i in range(M):
    for j in range(N):
        magnitude_spectrum[i][j] = int(round(abs(dft_matrix[i][j])))

# finding the min and max for scaling
min = np.amin(magnitude_spectrum)
max = np.amax(magnitude_spectrum)

# scaling the magnitude spectrum
for i in range(M):
    for j in range(N):
        magnitude_spectrum[i][j] = ((magnitude_spectrum[i][j]-min)/(max-min))*255

# displaying it
img = Image.fromarray(magnitude_spectrum)
img.show()

# the centered 2d dft of image
centered_2d_dft = np.ones((M,N))*0

# computing the centered dft by multiplying with (-1)^(n+m)
for i in range(M):
    for j in range(N):
        centered_2d_dft[i][j] = padded_matrix[i][j]*pow(-1,i+j)

# finding the dft
dft_image_matrix = np.fft.fft2(centered_2d_dft)

# for plotting the magnitude spectrum
magnitude_spectrum = np.ones((M,N))*0

# filling the values of magnitude spectrum
for i in range(M):
    for j in range(N):
        magnitude_spectrum[i][j] = int(round(abs(dft_image_matrix[i][j])))

# finding the min and max value in the matrix for scaling
max = np.amax(magnitude_spectrum)
min = np.amin(magnitude_spectrum)

# scaling the pixel values
for i in range(M):
    for j in range(N):
        magnitude_spectrum[i][j] = ((magnitude_spectrum[i][j]-min)/(max-min))*255

# display the magnitude spectrum
img = Image.fromarray(magnitude_spectrum)
img.show()

# defining the filter
filter = np.ones((2*M,2*N))*1
filter_to_show = np.ones((2*M,2*N))*255

# def make_filter(filter,x,y):
#     filter[x-1][y-1] = 0
#     filter[x-1][y] = 0
#     filter[x-1][y+1] = 0
#     filter[x][y-1] = 0
#     filter[x][y] = 0
#     filter[x][y+1] = 0
#     filter[x+1][y-1] = 0
#     filter[x+1][y] = 0
#     filter[x+1][y+1] = 0
#     return filter

# filter = make_filter(filter,33,33)
# filter = make_filter(filter,223,223)

# defining D(u,v)
def Duv(i,j,c):
    i = pow(i-c,2)
    j = pow(j-c,2)
    return pow(i+j,0.5)

# creating the filter for the image
for i in range(2*M):
    for j in range(2*N):
        duv1 = Duv(i,j,192)
        duv2 = Duv(i,j,320)
        if duv1<=3:
            filter[i][j] = 0
            filter_to_show[i][j] = 0
        if duv2<=3:
            filter[i][j] = filter[i][j]*0
            filter_to_show[i][j] = filter_to_show[i][j]*0

# displaying the filter
img = Image.fromarray(filter_to_show)
img.show()

# taking elementwise multiplication
elementwise_multiplied = np.multiply(dft_matrix,filter)

# computing the inverse dft
idft = np.fft.ifftn(elementwise_multiplied)

# taking the real part of the matrix
real_idft = idft.real

# final image
img = Image.fromarray(real_idft)
img.show()