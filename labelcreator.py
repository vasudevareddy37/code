import os
import numpy as np
# import cv2 as cv
# import timeit


inputFolder = '/home/mainampati/thesis/Emo_db'
suffix = '.png'

filename  = os.listdir(inputFolder)
L = len(filename)
labels = np.zeros((L,))
print"shape of the labels..." + str(labels.shape) 

for i in range (0, L):

    base_filename = filename[i]
    #print(base_filename)
    #name = os.path.join(inputFolder, base_filename)
    #print(base_filename[5])
    if base_filename[5] == 'A':
        #labels = np.r_[labels, 1]
        labels[i] = 1
    
    elif base_filename[5] == 'E':
        #labels = np.r_[labels, 2]
        labels[i] = 2
    elif base_filename[5] == 'F':
        #labels = np.r_[labels, 3]
        labels[i] = 3
    elif base_filename[5] == 'L':
        #labels = np.r_[labels, 4]
        labels[i] = 4
    elif base_filename[5] == 'N':
        #labels = np.r_[labels, 5]
        labels[i] = 5
    elif base_filename[5] == 'T':
        #labels = np.r_[labels, 6]
        labels[i] = 6
    elif base_filename[5] == 'W':
        #labels = np.r_[labels, 6]
        labels[i] = 7
    else:
        #labels = np.r_[labels, 7]
        print "problem with the file %s " % base_filename

y = np.copy(labels)

np.save('/home/mainampati/thesis/y_berlin_o.npy', y)
