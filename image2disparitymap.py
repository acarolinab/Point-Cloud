import numpy as np
import cv2
from matplotlib import pyplot as plt
import argparse


ap = argparse.ArgumentParser( description='Image RGB')
ap.add_argument("-l", 
    "--imageL", required=True,
	help="path to the left image file")
ap.add_argument("-r",
    "--imageR", required=True,
    help="path to the right image file")
args = vars(ap.parse_args())


imgL = cv2.imread(args['imageL'],0)
imgR = cv2.imread(args['imageR'],0)



""" stereo = cv2.StereoBM_create(numDisparities=16, blockSize=5)

disparity = stereo.compute(imgL,imgR)

plt.imshow(disparity,'gray')

plt.show()  """

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
disparity_map = stereo.compute(imgL, imgR)

plt.imshow(disparity_map,'gray')
plt.show()
