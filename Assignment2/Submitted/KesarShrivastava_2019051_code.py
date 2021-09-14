"""
Name: Kesar Shrivastava
Roll number: 2019051
"""

# Import the necessary libraries
from math import *
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

# Q3
print()
print("Solution to 3 goes here: ")

# input the path of the image
print()
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

# Q4
print()
print("Solution to 4 goes here: ")

# input the path of the image
print()
path = input("Enter the path of the image: ")

# load the image and convert into
# numpy array
img = Image.open(path)
img.show()

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

# matrix = matrix*(1.0)

# gamma transformatin
# c = 255/(max(np.amax(matrix))**(0.5))
max_val = np.amax(matrix)
gamma = 0.5
c = 255/(max_val**gamma)
# print(c)

gamma_matrix = c*(matrix**(0.5))

for i in range(256):
    for j in range(256):
        gamma_matrix[i][j] = round(gamma_matrix[i][j])

img_gamma = Image.fromarray(gamma_matrix)
img_gamma.show()

# Plotting normalised histogram of the target image
gg = np.ndarray.flatten(np.array(gamma_matrix))
plt.hist(gg, bins = 256, density = True, label = "Normalised Histogram")
plt.legend()
plt.xlabel("Target Image Intensity Level")
plt.ylabel("Normalised Value of Occurence")
plt.title("Histogram of Target Image")
plt.show()

# Plotting cdf for target image
count, bins_count = np.histogram(gg, bins=256)
pdf = count / sum(count)
cdf = np.cumsum(pdf)
plt.plot(bins_count[1:], cdf,".g", label="CDF (target)")
plt.xlabel("Target Image Intensity Level")
plt.ylabel("G(s)")
plt.title("CDF of Target Image")
plt.legend()
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

for i in range(1,256):
    pk_input[i] = pk_input[i]+pk_input[i-1]

nk_gamma = []
pk_gamma = {}
total = 65536

# Iterate for each pixel value
for px_val in range(256):
    indices = np.where(gamma_matrix==px_val)
    numOfIndices = np.size(indices)
    numOfIndices = numOfIndices/2
    nk_gamma.append(numOfIndices)
    pk_gamma[px_val] = numOfIndices/total

for i in range(1,256):
    pk_gamma[i] = pk_gamma[i]+pk_gamma[i-1]

mapped_values = {}

# print("pk_gamma: ", pk_gamma)

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
plt.hist(gg, bins = 256, density = True, label = "Normalised Histogram")
plt.legend()
plt.xlabel("Matched Image Intensity Level")
plt.ylabel("Normalised Value of Occurence")
plt.title("Histogram of Matched Image")
plt.show()

# Plotting cdf for matched image
count, bins_count = np.histogram(gg, bins=256)
pdf = count / sum(count)
cdf = np.cumsum(pdf)
plt.plot(bins_count[1:], cdf,".g", label="CDF (matched)")
plt.xlabel("Mathced Image Intensity Level")
plt.ylabel("F(r)")
plt.title("CDF of Matched Image")
plt.legend()
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

# Q5
print()
print("Solution to 5 goes here: ")
print()

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

n = input("Enter the upper limit for range of values: ")
m = input("Enter the lower limit for range of values: ")
print()

# Defining the image and filter
array = np.random.randint(m, n, size=(3, 3))
filter = np.random.randint(m, n, size=(3,3))
# array = np.array([[0,0,0],[0,1,0],[0,0,0]])
# filter = np.array([[1,2,3],[4,5,6],[7,8,9]])

# Printing the filter
print("The original filter: ")
print(filter)
print()

# Padding the image matrix
padded_matrix = np.ones((5,5))*0

for x in range(1,4):
    for y in range(1,4):
        padded_matrix[x][y] = array[x-1][y-1]

# Rotating the filter
transpose(filter)
reverseColumns(filter)
transpose(filter)
reverseColumns(filter)

# Printing the rotated filter
print("The rotated filter: ")
print(filter)
print()

# Printing the image matrix
print("The input matrix: ")
print(array)
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
                pixel_value = pixel_value+filter[x][y]*list[index]
                index = index+1
        
        output_matrix[i][j] = pixel_value

print("The output matrix: ")
print(output_matrix)

# # Initialising rotated filter
# rotatedFilter = np.ones((3,3))*-1

# # Rotating the filter
# rotatedFilter[0][0] = filter[2][2]
# rotatedFilter[0][1] = filter[2][1]
# rotatedFilter[0][2] = filter[2][0]
# rotatedFilter[1][0] = filter[1][2]
# rotatedFilter[1][1] = filter[1][1]
# rotatedFilter[1][2] = filter[1][0]
# rotatedFilter[2][0] = filter[0][2]
# rotatedFilter[2][1] = filter[0][1]
# rotatedFilter[2][2] = filter[0][0]
