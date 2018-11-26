import os
import numpy as np
import cv2 as cv
import timeit


start = timeit.default_timer()

inputFolder = '/home/mainampati/thesis/Emo_db/'
suffix = '.png'
filename  = os.listdir(inputFolder)


# you will get maxnNo_keypoints from maxKeypoints.py file
maxNo_keypoints = 391

features = []
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

    #flat = des.flatten('C')
    #flat1 = flat[:, np.newaxis] . T
    # print(des.shape, flat1.shape)
    #features.append(des)
    # exracing the same length of keyponts on each file.
    
    short = des[0:maxNo_keypoints, :]
    short_flat = short.flatten('C')
    #short_flat1 = short_flat[:, np.newaxis].T
    short_flat2 = short_flat[np.newaxis, :]
    #print short_flat1.shape == short_flat2.shape
    final_features = np.r_[final_features, short_flat2]
    
# print "The length of the feature vector" + str(len(features))




X_orginal = np.copy(final_features[1:, :])

np.save('/home/mainampati/thesis/X_original_o', X_orginal)

#X_norm = np.linalg.norm(X_orginal, axis= 1, keepdims= True)
#X_scaled1 = X_orginal / X_norm
#X_scaled = preprocessing.scale(X_orginal)
#print(X.shape == (final_features[1:, :]).shape)
#print "Sample vector X shape..." + str(X_orginal.shape) + str(final_features[1:, :].shape)
#y = np.copy(labels)

#np.save('y_V', y)


stop = timeit.default_timer()

print "***** The excution time =  " + str(stop - start)
