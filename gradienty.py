import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('digit_0.jpg',0)

sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize= 3)


plt.imshow(sobely,cmap = 'gray')
#plt.title('Sobel Y') 
plt.xticks([]), plt.yticks([])
plt.show()



