import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
#import csv
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
import timeit
#import pickle
#import unicodecsv as csv
start = timeit.default_timer()

inputFolder = '/home/mainampati/speech_framework/imagedb/'
suffix = '.png'
# for filename in os.listdir(inputFolder):
#   print(filename)

filename  = os.listdir(inputFolder)
features = []
final_features = np.zeros((1, 200*32))
L = len(filename)
labels = np.zeros((L,))
print"shape of the labels..." + str(labels.shape) 
for i in range (0, L):
    base_filename = filename[i]
    name = os.path.join(inputFolder, base_filename)
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
    else:
        #labels = np.r_[labels, 7]
        labels[i] = 7
    
 

    # print(name)
    img = cv.imread(name,0)
    # Initiate ORB detector
    orb = cv.ORB_create()
    # find the keypoints with ORB
    kp = orb.detect(img,None)
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)

    flat = des.flatten('C')
    flat1 = flat[:, np.newaxis] . T
    # print(des.shape, flat1.shape)
    features.append(flat1)
    '''
    if i == len(filename)-1:
        k = len(features)
        mini_vector_length = [0] * k
        for i in range (0, k):
            mini_vector_length[i] = (features[i].shape[1] / 32)
        continue
    '''
    short = des[0:200, :]
    short_flat = short.flatten('C')
    short_flat1 = short_flat[:, np.newaxis].T
    final_features = np.r_[final_features, short_flat1]
    
#print(min(mini_vector_length))
print"The length of the feature vector" + str(len(features))
print"Label array shape" + str(labels.shape)



X_orginal = np.copy(final_features[1:, :])
np.save('X_original_N', X_orginal)
#X_norm = np.linalg.norm(X_orginal, axis= 1, keepdims= True)
#X_scaled1 = X_orginal / X_norm
#X_scaled = preprocessing.scale(X_orginal)
#print(X.shape == (final_features[1:, :]).shape)
print "Sample vector X shape..." + str(X_orginal.shape) + str(final_features[1:, :].shape)
y = np.copy(labels)

np.save('y_N', y)

'''
#print(y.shape == labels[1:,].shape)
print "Checking the labels vector shapes..." + str( y.shape) + str( labels.shape)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.30, random_state=42)
print X_train.shape, y_train.shape
print X_test.shape, y_test.shape
clf = svm.SVC()
clf.fit(X_train, y_train)
#print(clf.predict(X_test) == y_test)

print"Svm mean predict score "+ str( clf.score(X_test, y_test) * 100)

clf1 = GaussianNB()
clf1.fit(X_train, y_train)
print"Navies mean predict score " + str(clf1.score( X_train, y_train) * 100)
'''
stop = timeit.default_timer()

print"Total excution time " + str( stop - start) + "seconds"

