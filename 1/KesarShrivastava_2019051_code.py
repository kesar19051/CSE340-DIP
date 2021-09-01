"""
Name: Kesar Shrivastava
Roll number: 2019051
"""

# Import the necessary libraries
from math import *
from PIL import Image
import numpy as np

print()
# Question 3 Solution
print("Solution to 3 goes here")
print()

# input the path of the image
path = input("Enter the path of the image: ")

# load the image and convert into
# numpy array
img = Image.open(path)
matrix = np.asarray(img)

# matrix = np.array([[5,10],[10,20]])
# matrix = np.array([[1,4,7],[10,13,16],[19,22,25]])

# define the interpolation factor
c = input("Enter the interpolation factor: ")
c = float(c)
# c = 2

# Show the input image
img.show()

# Print the entered interpolation factor
print("The interpolation factor is: ", end = "")
print(c)

# Define the number of rows and columns
M1 = len(matrix)
N1 = len(matrix[0])

padded_matrix = np.ones((M1+1,N1+1))*0

# pad with 0s for the last row and the last column 
for i in range(M1):
	for j in range(N1):
		padded_matrix[i][j] = matrix[i][j]

# Create the output matrix
M2 = round(c*(M1))
N2 = round(c*(N1))

new_matrix = np.ones((M2, N2))*-1

#filling in the output matirx
for i in range(M2):
	for j in range(N2):
		if new_matrix[i][j]==-1:
			x = i/c
			y = j/c

			if ceil(x)!=x:
				x1 = floor(x)
				x2 = ceil(x)
			else:
				if x==0:
					x1 = 0
					x2 = 1
				else:
					x1 = x-1
					x2 = x

			if ceil(y)!=y:
				y1 = floor(y)
				y2 = ceil(y)
			else: 
				if y==0:
					y1 = 0
					y2 = 1
				else:
					y1 = y-1
					y2 = y

			x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)

			X = [
					[x1,y1,x1*y1,1],
					[x1,y2,x1*y2,1],
					[x2,y2,x2*y2,1],
					[x2,y1,x2*y1,1],
				]

			Y = [
					[padded_matrix[x1][y1]],
					[padded_matrix[x1][y2]],
					[padded_matrix[x2][y2]],
					[padded_matrix[x2][y1]],
				]

			A = np.dot(np.linalg.pinv(X), Y)

			new_matrix[i][j] = np.dot(np.array([x,y,x*y,1]),A)

# Form the image and show
img_new = Image.fromarray(new_matrix)
img_new.show()

print("Q3 done")

print()

# Question 4 Solution
print("Solution to 4 goes here")
print()

# input the path of the image
path = input("Enter the path of the image: ")

# load the image and convert into
# numpy array
img = Image.open(path)
matrix = np.asarray(img)

# matrix = np.array([[5,10],[10,20]])

# rows and columns of the input matrix
M1 = len(matrix)
N1 = len(matrix[0])

# padding the matrix with zeroes to handle the corner cases
padded_matrix = np.ones((M1+1,N1+1))*0

for i in range(M1):
	for j in range(N1):
		padded_matrix[i][j] = matrix[i][j]

# defining rows and columns for the output matrix
M2 = 8*M1
N2 = 8*N1

# creating the output matrix
output_matrix = np.ones((M2,N2))*-1

# transformation_matrix = np.array([[1.414,-1.414,0],[1.414,1.414,0],[30,30,1]])
# transformation_matrix = np.array([[0.707,-0.707,0],[0.707,0.707,0],[0,0,1]])
# transformation_matrix = np.array([[1,0,0],[0,1,0],[30,50,1]])
# transformation_matrix = np.array([[0.52,-0.52,0],[0.52,0.52,0],[0,0,1]])

# input the transformation matrix
s1 = input("Enter the first row: ")
s2 = input("Enter the second row: ")
s3 = input("Enter the third row: ")

transformation_matrix = np.array([np.array(list(map(float, s1.split()))), 
								  np.array(list(map(float, s2.split()))),
	                              np.array(list(map(float, s3.split())))
	                             ])

# inverse of the transformaton matrix
inverse = np.linalg.pinv(transformation_matrix)

# filling in the output matrix
for i in range(M2):
	for j in range(N2):

		output = np.array([i-(4*M1),j-(4*N1),1])
		I = np.dot(output, inverse)
		x = I[0]
		y = I[1]

		if x<M1 and x>=0 and y<N1 and y>=0:

			if ceil(x)!=x:
				x1 = floor(x)
				x2 = ceil(x)
			else:
				if x==0:
					x1 = 0
					x2 = 1
				else:
					x1 = x-1
					x2 = x

			if ceil(y)!=y:
				y1 = floor(y)
				y2 = ceil(y)
			else: 
				if y==0:
					y1 = 0
					y2 = 1
				else:
					y1 = y-1
					y2 = y

			x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)

			# bilinear interpolation
			X = [
					[x1,y1,x1*y1,1],
					[x1,y2,x1*y2,1],
					[x2,y2,x2*y2,1],
					[x2,y1,x2*y1,1],
				]

			Y = [
					[padded_matrix[x1][y1]],
					[padded_matrix[x1][y2]],
					[padded_matrix[x2][y2]],
					[padded_matrix[x2][y1]],
				]

			A = np.dot(np.linalg.pinv(X), Y)

			output_matrix[i][j] = np.dot(np.array([x,y,x*y,1]),A)

# Display the input image
img.show()

# Form output image from the matrix
img_new = Image.fromarray(output_matrix)

# Print the transformation matrix
print()
print("The transformation matrix is ")
print(transformation_matrix)

# Display the output image
img_new.show()

img_new = img_new.convert("L")

# Save the output image
img_new.save("Q4_output.png")

print("Q4 done")

print()
# Question 5 solution
print("Solution to 5 goes here")
print()

# V is the reference image and X is the input

# first trial
V = np.array([[44,20,1],[42,8,1],[12,4,1]])
X = np.array([[121,-1,1],[102,-18,1],[55,19,1]])

# second trial
# V = np.array([[2,5,1],[7,2,1],[8,29,1]])
# X = np.array([[41,33,1],[43,24,1],[83,60,1]])

Z = np.dot(np.linalg.pinv(V),X)

# load input image and convert into matrix
input_img = Image.open('Q4_output.png')
input_matrix = np.asarray(input_img)
padded_matrix = np.ones((513,513))*0

# create padded matrix 
for i in range(512):
	for j in range(512):
		padded_matrix[i][j] = input_matrix[i][j]

output_matrix = np.ones((512,512))*-1

# filling in the output matrix
for i in range(512):
	for j in range(512):
		inputArr = np.array([i-256,j-256,1])
		output = np.dot(inputArr,Z)
		x = output[0]
		y = output[1]
		
		if(x<256 and x>=-256 and y<256 and y>=-256):
			if ceil(x)!=x:
				x1 = floor(x)
				x2 = ceil(x)
			else:
				if x==0:
					x1 = 0
					x2 = 1
				else:
					x1 = x-1
					x2 = x

			if ceil(y)!=y:
				y1 = floor(y)
				y2 = ceil(y)
			else: 
				if y==0:
					y1 = 0
					y2 = 1
				else:
					y1 = y-1
					y2 = y

			x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)

			# bilinear interpolation
			X = [
					[x1,y1,x1*y1,1],
					[x1,y2,x1*y2,1],
					[x2,y2,x2*y2,1],
					[x2,y1,x2*y1,1],
				]

			Y = [
					[padded_matrix[x1+256][y1+256]],
					[padded_matrix[x1+256][y2+256]],
					[padded_matrix[x2+256][y2+256]],
					[padded_matrix[x2+256][y1+256]],
				]

			A = np.dot(np.linalg.pinv(X), Y)

			output_matrix[i][j] = np.dot(np.array([x,y,x*y,1]),A)

# Display the reference image
ref_image = Image.open('assign1.jpg')
ref_image.show()

# Display the input image
input_img.show()

# Print Z
print("Z ")
print(Z)

# Display the registered image
img_new = Image.fromarray(output_matrix)
img_new.show()

print("Q5 done")

