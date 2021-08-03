import numpy as np
import cv2
from matplotlib import pyplot as plt
import argparse

def new_size(img):
    width = 640
    height = 480
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized


ap = argparse.ArgumentParser( description='Image RGB')
ap.add_argument("-l", 
    "--imageL", required=True,
	help="path to the left image file")
ap.add_argument("-r",
    "--imageR", required=True,
    help="path to the right image file")
args = vars(ap.parse_args())


imgL = cv2.imread(args['imageL'])
imgR = cv2.imread(args['imageR'])
(h, w, d) = imgL.shape
print("w: {}, h: {}, d: {}".format(w, h, d))
print("Size:",imgL.size)

grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

""" plt.imshow(grayR)
plt.show() """


esquerda = new_size(grayL)
direita = new_size(grayR)

#cv2.imshow('Gray image', esquerda)
  
#cv2.waitKey(0)
#cv2.destroyAllWindows()

win_size = 5
min_disp = -1
max_disp = 63 #min_disp * 9
num_disp = max_disp - min_disp # Needs to be divisible by 16

#Block matching. 
stereo = cv2.StereoSGBM_create(minDisparity= min_disp,
numDisparities = num_disp,
blockSize = 5,
uniquenessRatio = 5,
speckleWindowSize = 5,
speckleRange = 5,
disp12MaxDiff = 1,
P1 = 8*3*win_size**2, #8*3*win_size**2
P2 = 32*3*win_size**2) #32*3*win_size**2)

#Compute disparity map
print ("\nComputing the disparity  map...")
disparity_map = stereo.compute(esquerda, direita)

raw_h = disparity_map.shape[0]
raw_w = disparity_map.shape[1]
a = np.empty((raw_h,raw_w), np.uint16)

median = cv2.medianBlur(disparity_map,5)
#blur = cv2.bilateralFilter(disparity_map,9,75,75)

map = cv2.resize(disparity_map, (3840,2160), interpolation = cv2.INTER_AREA)



#plt.imshow(disparity_map, cmap='plasma')
plt.imshow(blur, cmap='plasma')
plt.show()
