# Import the necessary libraries
from math import *
from PIL import Image
import numpy as np

# V is the reference image and X is the input

# first trial
# V = np.array([[44,20,1],[42,8,1],[12,4,1]])
# X = np.array([[121,-1,1],[102,-18,1],[55,19,1]])

# second trial
# V = np.array([[2,5,1],[7,2,1],[8,29,1]])
# X = np.array([[41,33,1],[43,24,1],[83,60,1]])

# third trial
# V = np.array([[41,21,1],[13,5,1],[8,29,1]])
# X = np.array([[120,0,1],[55,20,1],[82,60,1]])


img = Image.open('assign1.jpg')
x1 = np.asarray(img)
	
# x1 = np.random.rand(103, 53)
	
plt.title('matplotlib.pyplot.ginput() function\
Example', fontweight ="bold")

print("After 3 clicks :")
plt.imshow(x1)
x = plt.ginput(3)

V = np.array([[x[0][1],x[0,0],1],[x[1][1],x[1][0],1],[x[2,1],x[2][0],1]])

img = Image.open('Q4_output.png')
x1 = np.asarray(img)
	
# x1 = np.random.rand(103, 53)
	
plt.title('matplotlib.pyplot.ginput() function\
Example', fontweight ="bold")

print("After 3 clicks :")
plt.imshow(x1)
x = plt.ginput(3)

X = np.array([[x[0][1],x[0,0],1],[x[1][1],x[1][0],1],[x[2,1],x[2][0],1]])

Z = np.dot(np.linalg.pinv(V),X)
print(Z)

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
print()
print("Z ")
print(Z)

# Display the registered image
img_new = Image.fromarray(output_matrix)
img_new.show()