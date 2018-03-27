import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
# import csv
# import pickle
# import unicodecsv as csv

inputFolder = '/home/mainampati/speech_framework/newdb/imagedb/'
suffix = '.png'
# for filename in os.listdir(inputFolder):
#   print(filename)

filename  = os.listdir(inputFolder)
features = []
for i in range (0, len(filename)):
    base_filename = filename[i]
    name = os.path.join(inputFolder, base_filename)
    # print(name)
    img = cv.imread(name,0)
    # Initiate ORB detector
    orb = cv.ORB_create(nfeatures=500, WTA_K=3)
    # find the keypoints with ORB
    kp = orb.detect(img,None)
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    #retval = cv.ORB.getMaxFeatures(orb)
    flat = des.flatten('C')
    flat1 = flat[:, np.newaxis] . T
    # print(des.shape, flat1.shape)
    features.append(flat1)
    print(des.shape, flat1.shape)
  
    '''
    with open('new_features1.txt', 'w') as f:
        csvwriter = csv.writer(f, lineterminator = '\n')
        for val in flat1:
            csvwriter.writerow([val])
        #csvwriter.writerows(flat1)
        f.close()
   
    with open('new_features.txt', 'a') as f:
        np.savetxt(f, flat1, delimiter=',', newline='\n')
        f.close()
    '''




print(len(features))

