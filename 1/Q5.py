# Import the necessary libraries
from math import *
from PIL import Image
import numpy as np

# load the image and convert into
# numpy array
img = Image.open('test.png')
matrix = np.asarray(img)

print(matrix)