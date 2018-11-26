import timeit
#import collections
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, recall_score
from sklearn.svm import SVC


## Starting timer
start = timeit.default_timer()

NUM_TRIALS = 1

## Loading the  dataset
x = np.load("/home/mainampati/thesis/X_original_o.npy")
y = np.load("/home/mainampati/thesis/y_berlin_o.npy")
group = np.load("/home/mainampati/thesis/group.npy") 

# analysing gruop data
#unique, counts = np.unique(group, return_counts=True)
#print(dict(zip(unique, counts)))
#print collections.Counter(group)

## preprocessing the trinanig data
x_scaled = x / 255

# Set up possible values of parameters to optimize over
p_grid = {"C": [1],
          "gamma": [.01]}

# We will use a Support Vector Classifier with "rbf" kernel
svm = SVC(kernel="rbf")

# array to store scroe
nested_scores = np.zeros(NUM_TRIALS)

# Loop for each trial
for i in range(NUM_TRIALS):

    # Choose cross-validation techniques for the inner and outer loops,
    # independently of the dataset.
    # E.g "GroupKFold", "LeaveOneOut", "LeaveOneGroupOut", etc.
    #inner_cv = KFold(n_splits=4, shuffle=True, random_state=i)
    loop_count = 0
    logo = LeaveOneGroupOut()
    #outer_cv = logo.split(x_scaled, y, group)
    print "number of outer splits %s" % logo.get_n_splits(groups = group)
    
    for trainval, testval in logo.split(x_scaled, y, group):
        groupval = group[trainval]
        x_inner_train = x_scaled[trainval]
        y_inner_train = y[trainval]
        inner_cv = logo.split(x_scaled[trainval], y[trainval], groupval)
        print "number of inner splits %s" % logo.get_n_splits(groups = groupval)
        #inner_cv = logo.split(groups=groupval)
        clf = GridSearchCV(estimator=svm, param_grid=p_grid, cv=inner_cv, scoring='recall_macro')
        clf.fit(x_scaled[trainval], y[trainval])

        print("Best parameters set found on train set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on val set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
                  
        print("Detailed classification report:")
        print()
        print("The model is trained on the full trainval set.")
        print("The scores are computed on the full testval set.")
        print()
        y_true, y_pred = y[testval], clf.predict(x_scaled[testval])
        print(classification_report(y_true, y_pred))
        print()
        #x_inner_train = x_scaled[trainval]
        #y_inner_train = y[trainval]
        loop_count += 1
        print "I am runnig %s loop" % loop_count
        raw_input("Press Enter to continue...")
stop = timeit.default_timer()
print("***** total excution time =  " + str((stop - start) / 60) + " minutes" )
















