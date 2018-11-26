from __future__ import print_function

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, recall_score
from sklearn.svm import SVC
import timeit
import numpy as np

start = timeit.default_timer()

print(__doc__)

# Loading the  dataset

x = np.load("/home/mainampati/thesis/X_original_o.npy")
y = np.load("/home/mainampati/thesis/y_berlin_o.npy")
group = np.load("/home/mainampati/thesis/group.npy") 

x_scaled = x / 255
print("number of keyponts considerd...." + str(
    x_scaled.shape[1] / 32))

# getting the toatl number of examples for the use of indices  
no_examples = x.shape[0]
indices = np.arange(no_examples)


# spliting the dataset into 2 parts along with the indices 
# to get group indices for the next step
x, x_h, y, y_h, idx1, idx2 = train_test_split(
    x_scaled, y, indices, test_size=0.2, random_state=42)
print("shape of training vector array : %r and train-label array shape : %r" %(
    x.shape, y.shape))

group_split = group[idx1]
print("shape of test vector array: %r and test-label array shape : %r" %(
    x_h.shape, y_h.shape))

# Initializing the LOGO method and getting no.of 
# splits according to data
logo = LeaveOneGroupOut()
no_splits = logo.get_n_splits(x, y, group_split) 
print("Number of splits..%d", no_splits)
cv = logo.split(x, y, group_split)

#{'Kernel': ['rbf'], 'gamma':[1e-3, 1e-4],'c': [0.01, 0.3, 1, 3, 10]},

#tuned_parameters = [{'kernel': ['rbf'], 'gamma':[1e-3, 1e-4], 'C': [0.01, 0.3, 1, 3, 10]}, {'kernel': ['linear'], 'C': [0.01, 0.3, 1, 3, 10]}]
tuned_parameters = [{'kernel': ['linear'], 'C': [0.01]}]

scores = ['recall'] 


for score in scores:
    print("# Tuning hyper-params for %s" % score)
    print()

    clf = GridSearchCV(SVC(), tuned_parameters, cv=cv,
    scoring='%s_macro' % score)

    clf.fit(x, y) 

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_h, clf.predict(x_h)
    #print(classification_report(y_true, y_pred))
    print(recall_score(y_true, y_pred, average= 'macro'))
    print()

stop = timeit.default_timer()
print("***** total excution time =  " + str((stop - start) / 60) + " minutes" )



