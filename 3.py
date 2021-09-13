"""
Name: Kesar Shrivastava
Roll number: 2019051
"""

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
img.show(title = "Input Image")
matrix = np.asarray(img)

# Plotting normalised histogram of the input image
gg = np.ndarray.flatten(np.array(matrix))
plt.hist(gg, bins = 256, density = True, label = "Normalised Histogram")
plt.legend()
plt.xlabel("Input Intensity Level")
plt.ylabel("Normalised Value of Occurence")
plt.title("Histogram of Input Image")
plt.show()

# Plotting cdf for input image
count, bins_count = np.histogram(gg, bins=256)
pdf = count / sum(count)
cdf = np.cumsum(pdf)
plt.plot(bins_count[1:], cdf,".g", label="CDF (input)")
plt.xlabel("Input Intensity Level")
plt.ylabel("H(r)")
plt.title("CDF of Input Image")
plt.legend()
plt.show()

nk = []
pk = {}
total = 65536
# Iterate for each pixel value
for px_val in range(256):
    indices = np.where(matrix==px_val)
    numOfIndices = np.size(indices)
    numOfIndices = numOfIndices/2
    nk.append(numOfIndices)
    pk[px_val] = numOfIndices/total

for i in range(1,256):
    pk[i] = pk[i]+pk[i-1]

for i in range(256):
    pk[i] = 255*pk[i]

# print(pk)
for i in range(256):
    pk[i] = round(pk[i])

new_matrix = np.ones((256,256))*-1
# print(pk)

for i in range(256):
    indices = np.where(matrix==i)
    size = np.size(indices)/2
    for j in range(int(size)):
        x = indices[0][j]
        y = indices[1][j]

        new_matrix[x][y] = pk[i]

img_new = Image.fromarray(new_matrix)
img_new.show(title = "Equalized Image")

nk = []
# Iterate for each pixel value
for px_val in range(256):
    indices = np.where(new_matrix==px_val)
    numOfIndices = np.size(indices)
    numOfIndices = numOfIndices/2
    nk.append(numOfIndices)

# Plotting normalised histogram of equalised image
gg = np.ndarray.flatten(np.array(new_matrix))
plt.hist(gg, bins = 256, density = True, label="Normalised Histogram")
plt.legend()
plt.xlabel("Equalised Image Intensity Level")
plt.ylabel("Normalised Value of Occurence")
plt.title("Higtogram of Equalised Image")
plt.show()

# Plotting cdf for output image
count, bins_count = np.histogram(gg, bins=256)
pdf = count / sum(count)
cdf = np.cumsum(pdf)
plt.plot(bins_count[1:], cdf,".g", label="CDF (output)")
plt.legend()
plt.xlabel("Output Intensity Level")
plt.ylabel("G(s)")
plt.title("CDF of Equalised Image")
plt.show()