import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
import timeit

start = timeit.default_timer()
X = np.load('/home/mainampati/speech_framework/X_original_A.npy')
y = np.load('/home/mainampati/speech_framework/y_A.npy')

X_scaled = X / 255
print("number of keyponts considerd...." + str(
    X_scaled.shape[1] / 32))

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=4)

print"shape of training vector array : %r and train-label array shape : %r" %(
    X_train.shape, y_train.shape)

print"shape of test vector array: %r and test-label array shape : %r" %(
    X_test.shape, y_test.shape)
       
print "train and test split percentages are.. %s & %s  "  %(
    ((float(X_train.shape[0]) / X_scaled.shape[0])* 100.0), (
        (float(X_test.shape[0]) / X_scaled.shape[0])* 100.0))


k = [0.001, 0.003, 0.01, 0.3, 1, 3, 10, 30]
clf = svm.SVC()
for i in range (0, len(k)):
    C = k[i]
    clf = svm.SVC(C = C, kernel = 'poly', degree= 3,
    decision_function_shape= 'ovr')
    clf.fit(X_train, y_train)
    print"Non-linear Svm mean predict score on test set " + str( clf.score(X_test, 
    y_test) * 100)
print "\n"
#clf.fit(X_train, y_train)
#print(clf.predict(X_test) == y_test)

'''
c = [0.001, 0.003, 0.01, 0.3, 1, 3]
## predicting on training set.
for i in range (0, len(c)):
    C = c[i]
    lin_clf = svm.LinearSVC(C = C)
    lin_clf.fit(X_train, y_train)
    print"LinearSvm mean predict score on tarin set.... " + str( 
    lin_clf.score(X_train, y_train) * 100)   
'''

c = [0.001, 0.003, 0.01, 0.3, 1, 3]
## prediction on test set.
for i in range (0, len(c)):
    C = c[i]
    lin_clf = svm.LinearSVC(C = C)
    lin_clf.fit(X_train, y_train)
    print"LinearSvm mean predict score on test set..... " + str( 
    lin_clf.score(X_test, y_test) * 100)

stop = timeit.default_timer()
print "***** total excution time =  " + str(stop - start)

