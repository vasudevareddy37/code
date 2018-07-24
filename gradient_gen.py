import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import timeit

start = timeit.default_timer()

inputFolder = '/home/mainampati/speech_framework/imagedb/'
suffix = '.png'
filename  = os.listdir(inputFolder)
L = len(filename)

for i in range (0, L):
    base_filename = filename[i]
    name = os.path.join(inputFolder, base_filename)

    img = cv.imread(name,0)
    sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=5)
    plt.xticks([]), plt.yticks([])
    plt.imshow(sobely,cmap = 'gray')
    plt.show



