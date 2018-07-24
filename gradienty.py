import numpy as np
import cv2 as cv
import pylab
from matplotlib import pyplot as plt

img = cv.imread('03a01Fa.png',0)

#sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize= 5)
sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
#laplacian = cv.Laplacian(img, cv.CV_64F)

fig = plt.figure(frameon = False)
plt.imshow(sobelx, cmap = 'gray')
#plt.title('Sobel Y') 
plt.xticks([]), plt.yticks([])
#plt.show()
fig.savefig('03a01Fay.png', bbox_inches= 'tight', pad_inches=0)



