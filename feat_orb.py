import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('03a01Nc.png',0)
# Initiate ORB detector
orb = cv.ORB_create(nfeatures=500, WTA_K=4)
#nfeatures = cv.ORB.setMaxFeatures(orb,300)
# find the keypoints with ORB
kp = orb.detect(img,None)
# compute the descriptors with ORB
kp, des = orb.compute(img, kp)
print( des.shape )
# draw only keypoints location,not size and orientation
img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
plt.imshow(img2),plt.xticks([]), plt.yticks([]) 
plt.show()
#nfeatures = cv.ORB.setMaxFeatures(orb,300)

flat = des.flatten('C')
flat1 = flat[:, np.newaxis]

fgh = [flat1, flat1]