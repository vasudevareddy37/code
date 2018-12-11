# def maxKeypoints(inputFolder):
import os
#import numpy as np
import cv2 as cv
import timeit

start = timeit.default_timer()

inputFolder = '/home/mainampati/thesis/emo_db/spectro_db3'
suffix = '.png'
filename = os.listdir(inputFolder)
features = []
L = len(filename)
for i in range(0, L):
    base_filename = filename[i]
    name = os.path.join(inputFolder, base_filename)

    img = cv.imread(name, 0)
    # Initiate FAST detector
    star = cv.xfeatures2d.StarDetector_create()
    # Initiate BRIEF extractor
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
    # find the keypoints with STAR
    kp = star.detect(img,None)
    # compute the descriptors with BRIEF
    kp, des = brief.compute(img, kp)
    features.append(des.shape[0])
# finding the min features length in database
# to get same length vector to train the svm
print min(features)
print max(features)
stop = timeit.default_timer()

print "***** The excution time =  " + str(stop - start)

#return features
