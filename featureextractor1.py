#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 00:27:12 2018

@author: mainampati
"""

import os
import numpy as np
import cv2 as cv
import timeit


start = timeit.default_timer()


mainFolder = '/home/mainampati/thesis/emodbupd/'
resultFolder = '/home/mainampati/thesis/emodbupdRe/'

numkeypts = [100, 150, 200, 250, 391]
forName = ['A', 'B', 'C', 'D', 'E']

totalSubfolds = os.listdir(mainFolder)
totalSubfolds.sort()


for t in range (0, len(totalSubfolds)):
    print(" hey i am working on %s loop out of upper %s loops" %
          ((t+1), len(totalSubfolds)))
    
    
    
    subfold = totalSubfolds[t]
    inputFolder = os.path.join(mainFolder, subfold)
    suffix = '.png'
    filename  = os.listdir(inputFolder)
    
    for rname, keypoints in zip(forName, numkeypts):
        # you will get maxnNo_keypoints from maxKeypoints.py file
        maxNo_keypoints = keypoints
    
        #features = []
        final_features = np.zeros((1, maxNo_keypoints*32))
    
        L = len(filename)
    
        for i in range (0, L):
            base_filename = filename[i]
            name = os.path.join(inputFolder, base_filename)
            
                # print(name)
            img = cv.imread(name,0)
            # Initiate ORB detector
            orb = cv.ORB_create()
            # find the keypoints with ORB
            kp = orb.detect(img,None)
            # compute the descriptors with ORB
            kp, des = orb.compute(img, kp)

            # exracing the same length of keyponts on each file.
            
            short = des[0:maxNo_keypoints, :]
            short_flat = short.flatten('C')
            #short_flat1 = short_flat[:, np.newaxis].T
            short_flat2 = short_flat[np.newaxis, :]
            #print short_flat1.shape == short_flat2.shape
            final_features = np.r_[final_features, short_flat2]
        

        X_orginal = np.copy(final_features[1:, :])
        
        reName = os.path.join(resultFolder, (subfold + rname + '.npy'))
    
        np.save(reName, X_orginal)
        print(" hey i am done with %s keypoints " % rname)
    
    stop = timeit.default_timer()
    print("***** total  programm excution time = %0.3f min" % 
          ((stop - start) / 60.0))