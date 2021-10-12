# Import the necessary libraries
from math import *
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

# input the path of the image
path = 'camera.jpg'

# load the image and convert into
# numpy array
img = Image.open(path)
img.show()
matrix = np.asarray(img)