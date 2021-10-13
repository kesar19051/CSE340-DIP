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

dft = np.fft.fft2(padded_matrix)

# for plotting the magnitude spectrum
magnitude_spectrum = np.ones((2*M,2*N))*0

# filling the values of magnitude spectrum
for i in range(2*M):
    for j in range(2*N):
        magnitude_spectrum[i][j] = int(round(abs(dft[i][j])))

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
n = 2

def func(D0):

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

    centered_dft_filter = np.fft.fft2(filter)

    magnitude_spectrum = np.ones((2*M,2*N))*0

    for i in range(2*M):
        for j in range(2*N):
            magnitude_spectrum[i][j] = round(abs(centered_dft_filter[i][j]))

    max = np.amax(magnitude_spectrum)
    min = np.amin(magnitude_spectrum)

    for i in range(2*M):
        for j in range(2*N):
            magnitude_spectrum[i][j] = ((magnitude_spectrum[i][j]-min)/(max-min))*255
    
    img = Image.fromarray(magnitude_spectrum)
    img.show()

    centered__ = np.ones((2*M,2*N))*0

    for i in range(2*M):
        for j in range(2*N):
            centered__[i][j] = filter[i][j]*pow(-1,i+j)

    centered_dft_filter = np.fft.fft2(centered__)

    magnitude_spectrum = np.ones((2*M,2*N))*0

    for i in range(2*M):
        for j in range(2*N):
            magnitude_spectrum[i][j] = round(abs(centered_dft_filter[i][j]))

    max = np.amax(magnitude_spectrum)
    min = np.amin(magnitude_spectrum)

    for i in range(2*M):
        for j in range(2*N):
            magnitude_spectrum[i][j] = ((magnitude_spectrum[i][j]-min)/(max-min))*255
    
    img = Image.fromarray(magnitude_spectrum)
    img.show()

    # taking elementwise multiplication
    elementwise_multiplied = np.multiply(dft_image_matrix,filter)

    max = np.amax(elementwise_multiplied)
    min = np.amin(elementwise_multiplied)

    elementwise_multiplied_show = np.ones((2*M,2*N))*0

    for i in range(2*M):
        for j in range(2*N):
            elementwise_multiplied_show[i][j] = ((abs(elementwise_multiplied[i][j])-min)/(max-min))*255

    img = Image.fromarray(elementwise_multiplied_show)
    img.show()

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

# defining D(u,v)
def Duv(i,j):
    i = pow(i-256,2)
    j = pow(j-256,2)
    return pow(i+j,0.5)

D = [10,30,60]

for D0 in D:
    func(D0)