import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneGroupOut, KFold
from sklearn.model_selection import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import recall_score, accuracy_score
from sklearn import svm
import numpy as np
import timeit


start = timeit.default_timer()

x = np.load("/home/mainampati/thesis/X_original_o.npy")
y = np.load("/home/mainampati/thesis/y_berlin_o.npy")
group = np.load("/home/mainampati/thesis/group.npy")            

x_scaled = x / 255
print("number of keyponts considerd...." + str(
    x_scaled.shape[1] / 32))

# getting the toatl number of examples for the use of indices  
no_examples = x.shape[0]
indices = np.arange(no_examples)

x, x_h, y, y_h, idx1, idx2 = train_test_split(
    x_scaled, y, indices, test_size=0.2, random_state=42)
print "shape of training vector array : %r and train-label array shape : %r" %(
    x.shape, y.shape)

group_split = group[idx1]
print"shape of test vector array: %r and test-label array shape : %r" %(
    x_h.shape, y_h.shape)

#logo = LeaveOneGroupOut( )

#no_splits = logo.get_n_splits(x, y, group_split)

loso = KFold(n_splits=5)

#print("Number of splits..%d", no_splits)

'''
for train_index, test_index in logo.split(x, y, group_split):

    #print("Train:", train_index, "Test:", test_index)
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]


    print"shape of training vector array : %r and train-label array shape : %r" %(x_train.shape, y_train.shape)
    print"shape of test vector array: %r and test-label array shape : %r" %(x_test.shape, y_test.shape)
    print "train and test split percentages are.. %s & %s  "  %(
    ((float(x_train.shape[0]) / x_scaled.shape[0])* 100.0), (
        (float(x_test.shape[0]) / x_scaled.shape[0])* 100.0))
'''


#clf = svm.SVC(C = 1, kernel = 'poly',
 #   degree= 3,decision_function_shape= 'ovr')

clf = svm.LinearSVC(C = 0.01)

#cv = logo.split(x, y, group_split)
cv = loso.split(x)
scores = cross_val_score(clf, x, y, cv = cv)

print scores
print scores.mean()
print scores.std() * 100 

clf.fit(x, y)

print"Non-linear Svm mean predict score on test set " + str( clf.score(x_h, 
    y_h) * 100) 

y_true, y_pred = y_h, clf.predict(x_h)

print(recall_score(y_true, y_pred, average= 'macro'))
print(accuracy_score(y_true, y_pred))

stop = timeit.default_timer()
print("***** total excution time =  " + str((stop - start) / 60) + " minutes" )








