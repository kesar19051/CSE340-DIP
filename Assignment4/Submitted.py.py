import cv2
import numpy as np
from math import *
from PIL import Image
import matplotlib.pyplot as plt

print("Solution to Q1 goes here: ")

org_img = Image.open('image.jpg')
org_img.show()

# img_path = input("path: ")
img = Image.open("noiselm.jpg")
img.show()
img = np.asarray(img)

padded_img = np.zeros((512, 512))*0

for i in range(256):
    for j in range(256):
        padded_img[i][j] = img[i][j]

# padded_img[:256, :256] = img

img_ = Image.fromarray(padded_img)
img_.show()

# making box filter...
box_filter = np.ones((11, 11))

for i in range(11):
    for j in range(11):
        box_filter[i][j] = box_filter[i][j]/121

padded_box_filter = np.ones((512, 512))*0

for i in range(11):
    for j in range(11):
        padded_box_filter[i][j] = box_filter[i][j]

# making laplacian filter...
laplacian = np.array([[0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]])
padded_laplacian = np.ones((512, 512))*0

for i in range(3):
    for j in range(3):
        padded_laplacian[i][j] = laplacian[i][j]

F_noise_img = np.fft.fft2(padded_img)
F_box = np.fft.fft2(padded_box_filter)
F_laplacian = np.fft.fft2(padded_laplacian)

F_box_conjugate = np.conjugate(F_box)
F_box_abs_sq = np.square(np.abs(F_box))

F_laplacian_abs_sq = np.square(np.abs(F_laplacian))

lambdas = np.array([0, 0.25, 0.5, 0.75, 1])
lambda_optimal = inf

least_mse = inf
best_image = np.ones((256, 256))*0

def do():
	for l in lambdas:
		cls = np.divide(F_box_conjugate, (F_box_abs_sq + l*F_laplacian_abs_sq))
		cls_print = cls.astype(np.uint8)
		x = "Cls with lambda: "
		x = x + str(l)
		cv2.imshow(x, cls_print)
		F_restored = np.multiply(cls, F_noise_img)
		f_restored = np.fft.ifft2(F_restored).real
		f_restored_cropped = np.ones((256, 256))*0
		for i in range(256):
			for j in range(256):
				f_restored_cropped[i][j] = f_restored[i][j]

		mse = np.mean(np.square(org_img - f_restored_cropped))
		print(mse)
		if mse < least_mse:
			least_mse = mse
			lambda_optimal = l
			best_image = f_restored_cropped
	return best_image

# do()


for l in lambdas:
    cls = np.divide(F_box_conjugate, (F_box_abs_sq + l*F_laplacian_abs_sq))
    cls_print = cls.astype(np.uint8)
    x = "Cls with lambda: "
    x = x + str(l)
    cv2.imshow(x, cls_print)
    F_restored = np.multiply(cls, F_noise_img)
    f_restored = np.fft.ifft2(F_restored).real
    f_restored_cropped = np.ones((256, 256))*0
    for i in range(256):
        for j in range(256):
            f_restored_cropped[i][j] = f_restored[i][j]

    # mean square error

    mse = np.mean(np.square(org_img - f_restored_cropped))
    print(mse)
    if mse < least_mse:
        least_mse = mse
        lambda_optimal = l
        best_image = f_restored_cropped
# best_image = do()
restored_img = best_image.astype(np.uint8)
img_ = Image.fromarray(restored_img)
img_.show()

print("The optimal lambda: ")
print(str(lambda_optimal))
print("The least MSE: ")
print(str(least_mse))

PSNR = 10*log10((255*255)/least_mse)

print("PSNR: ")
print(str(PSNR))
print()

print("Solution to Q3 goes here")

rgb_matrix = cv2.imread("./lena.tif", 1)
cv2.imshow("Input Image", rgb_matrix)

rgb_matrix = np.float32(rgb_matrix)/255

Hue = np.ones((512, 512))*0
Saturation = np.ones((512, 512))*0
Intensity = np.ones((512, 512))*0

for i in range(512):
    for j in range(512):
        r = (rgb_matrix[i][j][0])
        g = (rgb_matrix[i][j][1])
        b = (rgb_matrix[i][j][2])

        val1 = r+b+g
        val2 = (r - g) * (r - g) + (r - b) * (g - b)

        if val1 == 0:
            val1 = val1 + 0.0001
        
        if val2 == 0:
            val2 = val2 + 0.0001
        val = ((r-g) + (r-b))/2*pow(val2,0.5)
        theta = acos(val)
        if b <= g:
            Hue[i][j] = theta
        else:
            Hue[i][j] = 2*pi - theta

        Saturation[i][j] = 1 - (3 * min(r, min(g, b)))/(val1)

        Intensity[i][j] = (r + g + b)/3

img = np.multiply(Intensity, 255)
img = np.uint8(img)

plt.hist(np.ndarray.flatten(np.array(img)), bins=256, density=True)
plt.title("Input Image Normalized Histogram")
plt.show()
n = 512
m = 512
Hi = [0]*256

for i in range(n):
    for j in range(m):
        index = int(img[i][j])
        Hi[index] += 1

for i in range(256):
    Hi[i] /= (n*m)

for i in range(1, 256):
    Hi[i] = Hi[i] + Hi[i - 1]

for i in range(256):
    Hi[i] *= 256

output = np.ones((n, m))*0

for i in range(n):
    for j in range(m):
        output[i][j] = round(Hi[int(img[i][j])])

plt.hist(np.ndarray.flatten(np.array(output)), bins=256, density=True)
plt.title("Output Image Normalized Histogram")
plt.show()
Final_I = output
Final_I = np.float32(Final_I)/255

HSI_image = cv2.merge((Hue, Saturation, Intensity))
cv2.imshow('HSI image', HSI_image)

R = np.ones((512, 512))*0
G = np.ones((512, 512))*0
B = np.ones((512, 512))*0

Final_Image = np.ones((512, 512, 3))*0

for i in range(512):
    for j in range(512):
        if 0 <= Hue[i][j] and Hue[i][j]< 2*pi/3:
            B[i][j] = Final_I[i][j] * (1 - Saturation[i][j])
            R[i][j] = Final_I[i][j] * (
                    1 + (Saturation[i][j] * cos(Hue[i][j]) / cos(pi/3 - Hue[i][j])))
            G[i][j] = 3 * Final_I[i][j] - (R[i][j] + B[i][j])
        if 2*pi/3 <= Hue[i][j] and Hue[i][j] < 4*pi/3:
            Hue[i][j] = Hue[i][j] - 2*pi/3
            R[i][j] = Final_I[i][j] * (1 - Saturation[i][j])
            G[i][j] = Final_I[i][j] * (
                    1 + (Saturation[i][j] * cos(Hue[i][j]) / cos(pi/3 - Hue[i][j])))
            B[i][j] = 3 * Final_I[i][j] - (R[i][j] + G[i][j])
        elif 4*pi/3 <= Hue[i][j] and Hue[i][j] < 2*pi:
            Hue[i][j] = Hue[i][j] - 4*pi/3
            G[i][j] = Final_I[i][j] * (1 - Saturation[i][j])
            B[i][j] = Final_I[i][j] * (
                    1 + (Saturation[i][j] * cos(Hue[i][j]) / cos(pi/3 - Hue[i][j])))
            R[i][j] = 3 * Final_I[i][j] - (G[i][j] + B[i][j])

        Final_Image[i][j][0] = R[i][j]
        Final_Image[i][j][1] = G[i][j]
        Final_Image[i][j][2] = B[i][j]

cv2.imshow('Final Image', Final_Image)
cv2.waitKey(0)
