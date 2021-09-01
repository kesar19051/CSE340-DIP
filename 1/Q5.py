# Import the necessary libraries
from math import *
from PIL import Image
import numpy as np

# load the image and convert into
# numpy array

# V = np.array([[44,20,1],[42,8,1],[12,4,1]])
# X = np.array([[121,-1,1],[102,-18,1],[55,19,1]])

V = np.array([[2,5,1],[7,2,1],[8,29,1]])
X = np.array([[41,33,1],[43,24,1],[83,60,1]])
# T_inverse = np.linalg.inv(np.array([[1.414,-1.414,0],[1.414,1.414,0],[30,30,1]]))

Z = np.dot(np.linalg.inv(V),X)
# Z[0][1] = 0-Z[0][1]
# Z[1][0] = 0-Z[1][0]
print(Z)

input_img = Image.open('test.png')
input_matrix = np.asarray(input_img)
padded_matrix = np.ones((513,513))*0

for i in range(512):
	for j in range(512):
		padded_matrix[i][j] = input_matrix[i][j]

output_matrix = np.ones((512,512))*-1

# new_matrix = np.ones((64,64))*-1

# for i in range(64):
# 	for j in range(64):
# 		new_matrix[i][j] = input_matrix[i+256][j+256]

for i in range(512):
	for j in range(512):
		inputArr = np.array([i,j,1])
		output = np.dot(inputArr,Z)
		x = output[0]
		y = output[1]
		
		if(x<512 and x>=0 and y<512 and y>=0):
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

img_new = Image.fromarray(output_matrix)
# img_new = img_new.convert("L")
# img_new.save("test__.png")
img_new.show()