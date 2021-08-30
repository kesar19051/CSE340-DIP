# Import the necessary libraries
from math import *
from PIL import Image
import numpy as np

# load the image and convert into
# numpy array
# img = Image.open('x5.bmp')
# matrix = np.asarray(img)
matrix = np.array([[5,10],[10,20]])

#define the interpolation factor
interpolationFactor = 2

M1 = len(matrix)
N1 = len(matrix[0])

print(matrix)

M2 = int(interpolationFactor*M1)
N2 = int(interpolationFactor*N1)

new_matrix = np.ones((M2, N2))*-1

#Create Output interpolated matrix structure
for i in range(M1):
	for j in range(N1):
		new_matrix[int(i*interpolationFactor)][int(j*interpolationFactor)] = matrix[i][j]


M_X = int(i*interpolationFactor)
M_Y = int(j*interpolationFactor)

print()

for i in range(M_X+1):
	for j in range(M_Y+1):
		if new_matrix[i][j]==-1:

			x = i/interpolationFactor
			y = j/interpolationFactor

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
					[matrix[x1][y1]],
					[matrix[x1][y2]],
					[matrix[x2][y2]],
					[matrix[x2][y1]],
				]

			A = np.dot(np.linalg.inv(X), Y)

			new_matrix[i][j] = np.dot(np.array([x,y,x*y,1]),A)

for i in range(M_X+1):
	for j in range(M_Y+1, len(new_matrix[0])):
		new_matrix[i][j] = new_matrix[i][j-1]

for j in range(len(new_matrix[0])):
	for i in range(M_X+1, len(new_matrix)):
		new_matrix[i][j] = new_matrix[i-1][j]

# img = Image.fromarray(new_matrix)
# # img.save('test.png')
# img.show()
# print("process ended")
print(new_matrix)


