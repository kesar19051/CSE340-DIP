"""
Name: Kesar Shrivastava
Roll number: 2019051
"""

# Import the necessary libraries
from math import *
from PIL import Image
import numpy as np

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

transformation_matrix = np.array([[1.414,-1.414,0],[1.414,1.414,0],[30,30,1]])
# transformation_matrix = np.array([[0.707,-0.707,0],[0.707,0.707,0],[0,0,1]])
# transformation_matrix = np.array([[1,2,3],[4,5,6],[7,8,9]])
# transformation_matrix = np.array([[0.52,-0.52,0],[0.52,0.52,0],[0,0,1]])

# input the transformation matrix
# s1 = input("Enter the first row: ")
# s2 = input("Enter the second row: ")
# s3 = input("Enter the third row: ")

# transformation_matrix = np.array([np.array(list(map(float, s1.split()))), 
# 								  np.array(list(map(float, s2.split()))),
# 	                              np.array(list(map(float, s3.split())))
# 	                             ])

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
