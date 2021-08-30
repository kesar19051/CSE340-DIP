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

output_matrix = np.ones((M1,N1))*-1

transformation_matrix = np.array([[2,0,0],[0,2,0],[0,0,1]])

inverse = np.linalg.inv(transformation_matrix)

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

			output_matrix[i][j] = np.dot(np.array([x,y,x*y,1]),A)

		

img = Image.fromarray(output_matrix)
# img.save('test.png')
img.show()
# print("Output")
# print(new_matrix)

print("process ended")