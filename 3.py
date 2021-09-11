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
img.show()
matrix = np.asarray(img)

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
    
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(nk, bins = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275])
 
count, bins_count = np.histogram(nk, bins = 256)
  
# finding the PDF of the histogram using count values
pdf = count / sum(count)

# using numpy np.cumsum to calculate the CDF
# We can also find using the PDF values by looping and adding
cdf = np.cumsum(pdf)

# Show plot
# plt.show()
# print(pk)
# Calculating the cumulative
for i in range(2,256):
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
img_new.show()
    
    

