# Import the necessary libraries
from math import *
from PIL import Image
import numpy as np

# load the image and convert into
# numpy array
img = Image.open('assign1.jpg')
matrix = np.asarray(img)
# matrix = np.array([[5,10],[10,20]])
# matrix = np.array([[1,4,7],[10,13,16],[19,22,25]])

#define the interpolation factor
c = 2

M1 = len(matrix)
N1 = len(matrix[0])

padded_matrix = np.ones((M1+1,N1+1))*0

for i in range(M1):
	for j in range(N1):
		padded_matrix[i][j] = matrix[i][j]

# print(matrix)
# print()
# print(padded_matrix)
# print()

M2 = round(c*(M1))
N2 = round(c*(N1))

new_matrix = np.ones((M2, N2))*-1

#Create Output interpolated matrix structure

# for i in range(M1):
# 	for j in range(N1):
# 		new_matrix[i*c][j*c] = matrix[i][j]


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

			A = np.dot(np.linalg.inv(X), Y)

			new_matrix[i][j] = np.dot(np.array([x,y,x*y,1]),A)

# img.save('test.png')

img = Image.fromarray(new_matrix)
img.show()
# print("Output")
# print(new_matrix)

# print("process ended")
# print(new_matrix)

