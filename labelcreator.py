import os
import numpy as np
# import cv2 as cv
# import timeit


inputFolder = '/home/mainampati/thesis/emo_db/spectro_db0'
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
        labels[i] = 1
        #labels[i] = 1
    
    elif base_filename[5] == 'E':
        labels[i] = 0 
        #labels[i] = 2
    elif base_filename[5] == 'F':
        labels[i] = 1
        # labels[i] = 3
    elif base_filename[5] == 'L':
        labels[i] = 0
        #labels[i] = 4
    elif base_filename[5] == 'N':
        labels[i] = 1
        #labels[i] = 5
    elif base_filename[5] == 'T':
        labels[i] = 0
        #labels[i] = 6
    elif base_filename[5] == 'W':
        labels[i] = 1
        #labels[i] = 7
    else:
        
        print "problem with the file %s " % base_filename

y = np.copy(labels)

outName = inputFolder + 'Re' + '/emodbArousal.npy'

np.save(outName, y)
