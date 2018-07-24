import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_score
from sklearn import svm, metrics
from sklearn.naive_bayes import GaussianNB
import timeit



start = timeit.default_timer()
X = np.load('/home/mainampati/spectro_db/XS_original_V.npy')
y = np.load('/home/mainampati/spectro_db/yS_V.npy')

X_scaled = X / 256
print(len(y))
print("number of keyponts considerd...." + str(
    X_scaled.shape[1] / 32))

#loo = LeaveOneOut()
lin_clf = svm.LinearSVC(C = 0.01)
#scoring = metrics.recall_score(y, average='sample')
scores = cross_val_score(lin_clf, 
    X_scaled, y, cv = 10, scoring= 'recall_micro')
#print(scores.mean())
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

stop = timeit.default_timer()

print "***** total excution time =  " + str((stop - start) / 60.0)
