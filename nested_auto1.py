#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 13:29:02 2018

@author: mainampati
"""

import timeit
import os
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneGroupOut
#from sklearn.metrics import recall_score
from sklearn import svm 


## Starting timer
start = timeit.default_timer()

inputFolder = '/home/mainampati/thesis/emo_db/spectrodbRe'
filename  = os.listdir(inputFolder)
filename.sort()

# remove and assign gruop from filenames
idx = filename.index('gruop.npy')
group_label = os.path.join(inputFolder, filename.pop(idx))
group = np.load(group_label)

ar_val = []

if 'emodbValence.npy' in filename:
    idx = filename.index('emodbValence.npy')
    ar_val.append(filename.pop(idx)) 

if 'emodbArousal.npy' in filename:
    idx = filename.index('emodbArousal.npy')
    ar_val.append(filename.pop(idx)) 

# Big dictionary to store values 
big_dict = {}

for name in ar_val:
    
    print("hey i started finding scores with %s labels" % name)
    labels_path = os.path.join(inputFolder, name)

    y = np.load(labels_path)
    
    # Small dictionary to store for each file name
    small_dict = {}

    # creating path names for training(X) data
    for name_x in filename:
        
        file_x = os.path.join(inputFolder, name_x)
        x = np.load(file_x)
        
        # preprocessing the trinanig data
        x_scaled = x / 255
        
        # We  are using a Support Vector Classifier with "rbf" kernel
        clf = svm.SVC(kernel = 'rbf', gamma= 0.01, C = 100)
       
        # using leave one groupout method
        logo = LeaveOneGroupOut()
        cv =  logo.split(x_scaled, y, groups=group)
        
        scores = cross_val_score(clf, x_scaled, 
                                 y, cv = cv, scoring= 'recall_macro')
        
        loopMean = scores.mean() 
        loopStd = scores.std()
        
        small_name_x = name_x[:-4]
        
        small_dict[small_name_x] = (loopMean, loopStd)
        
        print(" finished finiding scores with %s data" %name_x)

    big_name = name[5:12]

    big_dict[big_name] = small_dict
    
np.save('/home/mainampati/thesis/results3.npy', big_dict)   
    
stop = timeit.default_timer()
print("***** total  programm excution time = %0.3f min" % 
      ((stop - start) / 60.0))
































