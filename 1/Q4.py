# Import the necessary libraries
from math import *
from PIL import Image
import numpy as np

# load the image and convert into
# numpy array
img = Image.open('x5.bmp')
matrix = np.asarray(img)

M1 = len(matrix)
N1 = len(matrix[0])

# padding the matrix with zeroes to handle the corner cases
padded_matrix = np.ones((M1+1,N1+1))*0

for i in range(M1):
	for j in range(N1):
		padded_matrix[i][j] = matrix[i][j]


# creating the output matrix
output_matrix = np.ones((M1,N1))*-1

transformation_matrix = np.array([[1,0,0],[0,1,0],[0,0,1]])

inverse = np.linalg.inv(transformation_matrix)

# filling in the output matrix
for i in range(M1):
	for j in range(N1):

		output = np.array([i,j,1])
		I = np.dot(output, inverse)
		x = I[0]
		y = I[1]

		if x<M1 and y<N1:

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

			A = np.dot(np.linalg.inv(X), Y)

			output_matrix[i][j] = np.dot(np.array([x,y,x*y,1]),A)

		

img = Image.fromarray(output_matrix)
# img.save('test.png')
img.show()
# print("Output")
# print(new_matrix)

print("process ended")