# # Import the necessary libraries
# from math import *
# from PIL import Image
# import numpy as np
# from matplotlib import pyplot as plt
# from scipy import signal

# def q3(img):
#     # img = cv2.imread("input image.jpg", 0)
#     # print("Input image:")
#     # print(img)
#     # i = Image.fromarray(img)
#     # i.show()
#     plt.hist(np.ndarray.flatten(np.array(img)), bins=256, density=True)
#     plt.title("Input Image Normalized Histogram")
#     plt.show()
#     n, m = len(img), len(img[0])

#     H = []
#     for i in range(256):
#         H.append(0)

#     for i in range(n):
#         for j in range(m):
#             index = int(img[i][j])
#             H[index] += 1

#     for i in range(256):
#         H[i] /= (n*m)

#     for i in range(1, 256):
#         H[i] += H[i - 1]

#     for i in range(256):
#         H[i] *= 256

#     output = np.zeros((n, m))
#     for i in range(n):
#         for j in range(m):
#             output[i][j] = round(H[int(img[i][j])])

#     print("Output image:")
#     print(output)
#     # i = Image.fromarray(output)
#     # i.show()

#     plt.hist(np.ndarray.flatten(np.array(output)), bins=256, density=True)
#     plt.title("Output Image Normalized Histogram")
#     plt.show()

#     return output

# def convolution():
#     n1 = int( input( "Enter lower limit of range for 3x3 matrices" ) )
#     n2=int(input("Enter upper limit of range for 3x3 matrices"))
#     matrix = np.random.randint(low=n1, high=n2, size=(3, 3))
#     filter=array = np.random.randint( low=n1,high=n2, size=(3, 3))
#     # matrix=[[0,0,0],[0,1,0],[0,0,0]]
#     # filter=[[1,2,3],[4,5,6],[7,8,9]]
#     # matrix = np.array( [[5, 199, 153], [43, 251, 188], [178, 234, 56]] )
#     # filter = np.array( [[1, 196, 12], [77, 178, 120], [67, 20, 247]] )
#     print("Input matrix")
#     print(matrix)
#     print("filter")
#     print(filter)
#     padded_matrix=np.ones((5,5))*0
#     output_matrix = np.ones( (5, 5) ) * 0
#     for i in range(1,4):
#         for j in range(1,4):
#             padded_matrix[i][j]=matrix[i-1][j-1]
#     filter=np.rot90(filter,2)
#     print("Rotated filter")
#     print(filter)
#     for i in range(5):
#         for j in range(5):
#             new_value=0
#             for p in range(-1,2):
#                 for q in range(-1,2):
#                     x=i+p
#                     y=j+q
#                     if(x>=0 and x<5 and y>=0 and y<5):
#                         new_value+=padded_matrix[x][y]*filter[p+1][q+1]
#             output_matrix[i][j]=new_value
#     print("Output matrix")
#     print(output_matrix)

# def histogramMatching(image):
#     matrix=getImageMatrix(image)
#     print("input image matrix")
#     print(matrix)
#     makeNormalisedHistogram( matrix,"input image histogram" )
#     target_matrix=gammaTransformation(0.5,matrix)
#     print("target image matrix")
#     print(target_matrix)
#     displayImageFromMatrix(target_matrix)
#     makeNormalisedHistogram(target_matrix,"target image histogram")
#     H=getCdf(matrix,"Input image cdf")
#     G=getCdf(target_matrix,"target Image cdf")
#     print("H")
#     print(H)
#     print("G")
#     print(G)
#     mapping={}
#     for i in range(256):
#         mapping[i]=getClosestValueIndex(i,H,G)
#     matching_matrix=np.ones((len(matrix),len(matrix[0])))*0
#     for i in range(len(matrix)):
#         for j in range(len(matrix[0])):
#             matching_matrix[i][j]=mapping[matrix[i][j]]
#     displayImageFromMatrix(matching_matrix)
#     print("matching image matrix")
#     print(matching_matrix)
#     print("mapping")
#     print(mapping)
#     makeNormalisedHistogram(matching_matrix,"Matching Image Histogram")
# def getClosestValueIndex(i,cdf,cdf_target):
#     diff=1000000000000
#     index=0
#     for j in range(len(cdf_target)):
#         if abs(cdf[i]-cdf_target[j])<diff:
#             diff=abs(cdf[i]-cdf_target[j])
#             index=j
#     return index

# def getImageMatrix(image):
#     img = Image.open( image )
#     matrix = np.array( img )
#     img.show()
#     return matrix
# def gammaTransformation(gamma,matrix):
#     c = pow( 255, 1 - gamma )
#     output_matrix = np.ones( (len( matrix ), len( matrix[0] )) ) * 0
#     for i in range( len( matrix ) ):
#         for j in range( len( matrix[0] ) ):
#             r = matrix[i][j]
#             output_matrix[i][j] = round(c * pow( r, gamma ))
#     return output_matrix

# def displayImageFromMatrix(matrix):
#     data = Image.fromarray( matrix )
#     data.show()

# def histogramEqualisation(matrix):
#     # img = Image.open(image)
#     # matrix = np.array(img)
#     print("Input Image Matrix")
#     print(matrix)
#     getFrequencyOfElements(matrix)
#     makeNormalisedHistogram(matrix,"Input Image Histogram")
#     cdf=getCdf(matrix,"input image cdf")
#     print("Input image cdf")
#     print(cdf)
#     for i in range( len( matrix ) ):
#         for j in range( len( matrix[0] ) ):
#             matrix[i][j] = 255 * cdf[matrix[i][j]]
#     displayImageFromMatrix(matrix)
#     print("Output Image Matrix")
#     print(matrix)
#     makeNormalisedHistogram(matrix,"Output Image Histogram")
#     getFrequencyOfElements(matrix)
#     cdf=getCdf(matrix,"Output image cdf")
#     print( "Output image cdf" )
#     print( cdf )

# def getCdf(matrix,title):
#     gg = np.ndarray.flatten( np.array( matrix ) )
#     count, bins_count = np.histogram( gg, bins=256 )
#     pdf = count / sum( count )
#     cdf = np.cumsum( pdf )
#     plt.plot( bins_count[1:], pdf, color="red", label="PDF" )
#     plt.plot( bins_count[1:], cdf, label="CDF" )
#     plt.title(title)
#     plt.legend()
#     plt.show()
#     return cdf

# def makeNormalisedHistogram(matrix,title):
#     gg = np.ndarray.flatten( np.array( matrix ) )
#     plt.hist( gg, bins=256, density=True )
#     plt.title(title)
#     plt.show()

# def getFrequencyOfElements(matrix):
#     dict_frequency = {}
#     for i in range( 256 ):
#         dict_frequency[i] = len( matrix[np.where( matrix == i )] )
#     return dict_frequency

# # n=int(input("Enter question number"))
# # if n==3:
# #     img=input("Enter image name: ")
# #     # img = "inputimg.jpg"
# #     histogramEqualisation( img )
# # elif n==4:
# #     img = input( "Enter image name: " )
# #     # img = "inputimg.jpg"
# #     histogramMatching( img )
# # else:
# #      convolution()

# def Q3(img):

#     output_img = np.copy(img)
    
#     # row and col count
#     m = len(img)
#     n = len(img[0])
#     total_pixel = m * n

#     # Normalised histogram
#     h = []

#     for i in range(256):
#         y = np.where(img == i)
#         h.append(len(y[0]) / total_pixel)

#     # Finding CDF of histogram
#     cdf_sum = 0

#     H = []
#     for i in range(256):
#         cdf_sum = cdf_sum + h[i]
#         H.append(cdf_sum)
#     H = np.array(H)

#     # Constructing equalised image and its CDF
#     S = []
#     for i in range(256):
#         y = np.where(img == i)
#         S.append(255 * H[i])
#         output_img[y] = S[i]

#     # print(S)
#     output_img = output_img.astype(np.uint8)
#     s = []

#     for i in range(256):
#         y = np.where(output_img == i)
#         s.append(len(y[0]) / total_pixel)

#     # cv2.imshow("Input_Image", img)

#     # cv2.imshow("Output_Image", output_img)

#     f = plt.figure(1)
#     plt.bar([i for i in range(256)], h)
#     plt.xlabel("pixel val")
#     plt.ylabel("noramlised value for pixel")
#     plt.title("Histogram for Input Image")
#     plt.plot()
#     f.show()

#     g = plt.figure(2)
#     plt.bar([i for i in range(256)], s)
#     plt.xlabel("pixel val")
#     plt.ylabel("noramlised value for pixel")
#     plt.title("Histogram for Equalised Image")
#     g.show()

#     plt.show()

#     return output_img

# # input the path of the image
# path = 'lena.tif'

# # load the image and convert into
# # numpy array
# img = Image.open(path)
# # img.show()
# input_matrix = np.array(img)

# print(type(input_matrix[0][0][0]))

# #Hue
# hue = np.ones((512,512))*-1

# #saturation
# saturation = np.ones((512,512))*-1

# #intensity
# intensity = np.ones((512,512))*-1

# for i in range(512):
#     for j in range(512):

#         r = int(input_matrix[i][j][0])
#         g = int(input_matrix[i][j][1])
#         b = int(input_matrix[i][j][2])

#         intensity[i][j] = (r+g+b)/3
#         divByZero1 = r+g+b
#         if divByZero1==0:
#             divByZero1 = divByZero1+0.0001
#         saturation[i][j] = 1-(3*(min((min(r,g)),b)))/divByZero1
#         divByZero = (r-g)*(r-g)+(r-b)*(g-b)
#         if divByZero==0:
#             divByZero = divByZero+0.0001
#         theta = degrees(acos(((r-g + r-b)/2)/pow(divByZero,0.5)))

#         if b<=g:
#             hue[i][j] = theta
#         else:
#             hue[i][j] = 360-theta

# # intensity = intensity*255

# # img = img.fromarray(intensity)
# for i in range(512):
#     for j in range(512):
#         intensity[i][j] = int(round(intensity[i][j]))

# # histogramEqualisation(intensity)
# matrix = q3(intensity)

# # matrix = intensity

# answer = np.ones((512,512,3))*-1

# for i in range(512):
#     for j in range(512):

#         if 0<=hue[i][j] and hue[i][j]<120:
#             cosH = cos(radians(hue[i][j]))
#             cos60H = cos(radians(60-hue[i][j]))
#             b = matrix[i][j]*(1-saturation[i][j])
#             r = matrix[i][j]*(1+(saturation[i][j]*cosH)/cos60H)
#             g = 3*matrix[i][j]-(r+b)

#         if 120<=hue[i][j] and hue[i][j]<240:
#             hue[i][j] = hue[i][j]-120
#             cosH = cos(radians(hue[i][j]))
#             cos60H = cos(radians(60-hue[i][j]))
#             r = matrix[i][j]*(1-saturation[i][j])
#             g = matrix[i][j]*(1+(saturation[i][j]*cosH)/cos60H)
#             b = 3*matrix[i][j]-(r+g)

#         if 240<=hue[i][j] and hue[i][j]<360:
#             hue[i][j] = hue[i][j]-120
#             cosH = cos(radians(hue[i][j]))
#             cos60H = cos(radians(60-hue[i][j]))
#             g = matrix[i][j]*(1-saturation[i][j])
#             b = matrix[i][j]*(1+(saturation[i][j]*cosH)/cos60H)
#             r = 3*matrix[i][j]-(b+g)
        
#         answer[i][j][0] = int(round(r))
#         answer[i][j][1] = int(round(g))
#         answer[i][j][2] = int(round(b))

# img = Image.fromarray(answer, 'RGB')
# img.save('my.png')
# img.show()

# Import the necessary libraries
from math import *
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2
import pandas as pd


def q3(img):
    # img = cv2.imread("input image.jpg", 0)
    # print("Input image:")
    # print(img)
    # i = Image.fromarray(img)
    # i.show()
    plt.hist(np.ndarray.flatten(np.array(img)), bins=256, density=True)
    plt.title("Input Image Normalized Histogram")
    plt.show()
    n, m = len(img), len(img[0])

    H = []
    for i in range(256):
        H.append(0)

    for i in range(n):
        for j in range(m):
            index = int(img[i][j])
            H[index] += 1

    for i in range(256):
        H[i] /= (n * m)

    for i in range(1, 256):
        H[i] += H[i - 1]

    for i in range(256):
        H[i] *= 256

    output = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            output[i][j] = round(H[int(img[i][j])])

    print("Output image:")
    print(output)
    # i = Image.fromarray(output)
    # i.show()

    plt.hist(np.ndarray.flatten(np.array(output)), bins=256, density=True)
    plt.title("Output Image Normalized Histogram")
    plt.show()

    return output


def convolution():
    n1 = int(input("Enter lower limit of range for 3x3 matrices"))
    n2 = int(input("Enter upper limit of range for 3x3 matrices"))
    matrix = np.random.randint(low=n1, high=n2, size=(3, 3))
    filter = array = np.random.randint(low=n1, high=n2, size=(3, 3))
    # matrix=[[0,0,0],[0,1,0],[0,0,0]]
    # filter=[[1,2,3],[4,5,6],[7,8,9]]
    # matrix = np.array( [[5, 199, 153], [43, 251, 188], [178, 234, 56]] )
    # filter = np.array( [[1, 196, 12], [77, 178, 120], [67, 20, 247]] )
    print("Input matrix")
    print(matrix)
    print("filter")
    print(filter)
    padded_matrix = np.ones((5, 5)) * 0
    output_matrix = np.ones((5, 5)) * 0
    for i in range(1, 4):
        for j in range(1, 4):
            padded_matrix[i][j] = matrix[i - 1][j - 1]
    filter = np.rot90(filter, 2)
    print("Rotated filter")
    print(filter)
    for i in range(5):
        for j in range(5):
            new_value = 0
            for p in range(-1, 2):
                for q in range(-1, 2):
                    x = i + p
                    y = j + q
                    if (x >= 0 and x < 5 and y >= 0 and y < 5):
                        new_value += padded_matrix[x][y] * filter[p + 1][q + 1]
            output_matrix[i][j] = new_value
    print("Output matrix")
    print(output_matrix)


def histogramMatching(image):
    matrix = getImageMatrix(image)
    print("input image matrix")
    print(matrix)
    makeNormalisedHistogram(matrix, "input image histogram")
    target_matrix = gammaTransformation(0.5, matrix)
    print("target image matrix")
    print(target_matrix)
    displayImageFromMatrix(target_matrix)
    makeNormalisedHistogram(target_matrix, "target image histogram")
    H = getCdf(matrix, "Input image cdf")
    G = getCdf(target_matrix, "target Image cdf")
    print("H")
    print(H)
    print("G")
    print(G)
    mapping = {}
    for i in range(256):
        mapping[i] = getClosestValueIndex(i, H, G)
    matching_matrix = np.ones((len(matrix), len(matrix[0]))) * 0
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matching_matrix[i][j] = mapping[matrix[i][j]]
    displayImageFromMatrix(matching_matrix)
    print("matching image matrix")
    print(matching_matrix)
    print("mapping")
    print(mapping)
    makeNormalisedHistogram(matching_matrix, "Matching Image Histogram")


def getClosestValueIndex(i, cdf, cdf_target):
    diff = 1000000000000
    index = 0
    for j in range(len(cdf_target)):
        if abs(cdf[i] - cdf_target[j]) < diff:
            diff = abs(cdf[i] - cdf_target[j])
            index = j
    return index


def getImageMatrix(image):
    img = Image.open(image)
    matrix = np.array(img)
    img.show()
    return matrix


def gammaTransformation(gamma, matrix):
    c = pow(255, 1 - gamma)
    output_matrix = np.ones((len(matrix), len(matrix[0]))) * 0
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            r = matrix[i][j]
            output_matrix[i][j] = round(c * pow(r, gamma))
    return output_matrix


def displayImageFromMatrix(matrix):
    data = Image.fromarray(matrix)
    data.show()


def histogramEqualisation(matrix):
    # img = Image.open(image)
    # matrix = np.array(img)
    print("Input Image Matrix")
    print(matrix)
    getFrequencyOfElements(matrix)
    makeNormalisedHistogram(matrix, "Input Image Histogram")
    cdf = getCdf(matrix, "input image cdf")
    print("Input image cdf")
    print(cdf)
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix[i][j] = 255 * cdf[matrix[i][j]]
    displayImageFromMatrix(matrix)
    print("Output Image Matrix")
    print(matrix)
    makeNormalisedHistogram(matrix, "Output Image Histogram")
    getFrequencyOfElements(matrix)
    cdf = getCdf(matrix, "Output image cdf")
    print("Output image cdf")
    print(cdf)


def getCdf(matrix, title):
    gg = np.ndarray.flatten(np.array(matrix))
    count, bins_count = np.histogram(gg, bins=256)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    plt.plot(bins_count[1:], pdf, color="red", label="PDF")
    plt.plot(bins_count[1:], cdf, label="CDF")
    plt.title(title)
    plt.legend()
    plt.show()
    return cdf


def makeNormalisedHistogram(matrix, title):
    gg = np.ndarray.flatten(np.array(matrix))
    plt.hist(gg, bins=256, density=True)
    plt.title(title)
    plt.show()


def getFrequencyOfElements(matrix):
    dict_frequency = {}
    for i in range(256):
        dict_frequency[i] = len(matrix[np.where(matrix == i)])
    return dict_frequency


# n=int(input("Enter question number"))
# if n==3:
#     img=input("Enter image name: ")
#     # img = "inputimg.jpg"
#     histogramEqualisation( img )
# elif n==4:
#     img = input( "Enter image name: " )
#     # img = "inputimg.jpg"
#     histogramMatching( img )
# else:
#      convolution()

def Q3(img):
    output_img = np.copy(img)

    # row and col count
    m = len(img)
    n = len(img[0])
    total_pixel = m * n

    # Normalised histogram
    h = []

    for i in range(256):
        y = np.where(img == i)
        h.append(len(y[0]) / total_pixel)

    # Finding CDF of histogram
    cdf_sum = 0

    H = []
    for i in range(256):
        cdf_sum = cdf_sum + h[i]
        H.append(cdf_sum)
    H = np.array(H)

    # Constructing equalised image and its CDF
    S = []
    for i in range(256):
        y = np.where(img == i)
        S.append(255 * H[i])
        output_img[y] = S[i]

    # print(S)
    output_img = output_img.astype(np.uint8)
    s = []

    for i in range(256):
        y = np.where(output_img == i)
        s.append(len(y[0]) / total_pixel)

    # cv2.imshow("Input_Image", img)

    # cv2.imshow("Output_Image", output_img)

    f = plt.figure(1)
    plt.bar([i for i in range(256)], h)
    plt.xlabel("pixel val")
    plt.ylabel("noramlised value for pixel")
    plt.title("Histogram for Input Image")
    plt.plot()
    f.show()

    g = plt.figure(2)
    plt.bar([i for i in range(256)], s)
    plt.xlabel("pixel val")
    plt.ylabel("noramlised value for pixel")
    plt.title("Histogram for Equalised Image")
    g.show()

    plt.show()

    return output_img


# input the path of the image
path = 'lena.tif'
np.set_printoptions(threshold=np.inf)
# load the image and convert into
# numpy array
img = Image.open(path)
# img.show()
input_matrix = np.array(img)

print(type(input_matrix[0][0][0]))

# Hue
hue = np.ones((512, 512)) * -1

# saturation
saturation = np.ones((512, 512)) * -1

# intensity
intensity = np.ones((512, 512)) * -1

for i in range(512):
    for j in range(512):

        r = int(input_matrix[i][j][0])
        g = int(input_matrix[i][j][1])
        b = int(input_matrix[i][j][2])

        intensity[i][j] = (r + g + b) / 3
        divByZero1 = r + g + b
        if divByZero1 == 0:
            divByZero1 = divByZero1 + 0.0001
        saturation[i][j] = 1 - (3 * (min((min(r, g)), b))) / divByZero1
        divByZero = (r - g)*(r - g) + (r - b)*(g - b)
        if divByZero == 0:
            divByZero = divByZero + 0.0001
        theta = degrees(acos(((r - g + r - b) / 2) / pow(divByZero, 0.5)))

        if b <= g:
            hue[i][j] = theta
        else:
            hue[i][j] = 360 - theta

# intensity = intensity*255

# img = img.fromarray(intensity)
for i in range(512):
    for j in range(512):
        intensity[i][j] = int(round(intensity[i][j]))

# histogramEqualisation(intensity)
matrix = q3(intensity)

matrix = matrix - 1
np.savetxt('matrix.csv', matrix, delimiter=',')

np.savetxt('hue.csv', hue, delimiter=',')

np.savetxt('saturation.csv', saturation, delimiter=',')

# matrix = intensity

answer = np.ones((512, 512, 3)) * -1

for i in range(512):
    for j in range(512):
        if 0 <= hue[i][j] < 120:
            cosH = cos(radians(hue[i][j]))
            cos60H = cos(radians(60 - hue[i][j]))
            b = matrix[i][j] * (1 - saturation[i][j])
            r = matrix[i][j] * (1 + (saturation[i][j] * cosH) / cos60H)
            g = 3 * matrix[i][j] - (r + b)
        if 120 <= hue[i][j] < 240:
            hue[i][j] = hue[i][j] - 120
            cosH = cos(radians(hue[i][j]))
            cos60H = cos(radians(60 - hue[i][j]))
            r = matrix[i][j] * (1 - saturation[i][j])
            g = matrix[i][j] * (1 + (saturation[i][j] * cosH) / cos60H)
            b = 3 * matrix[i][j] - (r + g)

        if 240 <= hue[i][j] < 360:
            hue[i][j] = hue[i][j] - 120
            cosH = cos(radians(hue[i][j]))
            cos60H = cos(radians(60 - hue[i][j]))
            g = matrix[i][j] * (1 - saturation[i][j])
            b = matrix[i][j] * (1 + (saturation[i][j] * cosH) / cos60H)
            r = 3 * matrix[i][j] - (b + g)

        answer[i][j][0] = int(round(r))
        answer[i][j][1] = int(round(g))
        answer[i][j][2] = int(round(b))


Final_Image = answer.astype(np.uint8)

img = Image.fromarray(cv.cvtColor(answer, cv.COLOR_BGR2RGB), 'RGB')
img.show()
# img = Image.fromarray()

# cv.cvtColor(answer, cv.COLOR_BGR2RGB)
# cv2.imshow('color image', Final_Image)
# cv2.waitKey(0)