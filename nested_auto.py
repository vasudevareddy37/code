import timeit
import os
#import collections
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import  recall_score
from sklearn.svm import SVC


## Starting timer
start = timeit.default_timer()


## Loading the saved loads for dataset 
inputFolder = '/home/mainampati/thesis/test/'
filename  = os.listdir(inputFolder)
filename.sort()

# remove and assign gruop from filenames
idx = filename.index('gruop.npy')
group_label = os.path.join(inputFolder, filename.pop(idx))
group = np.load(group_label)

# creating lsit for arousal and valence files
ar_val = []
if 'emodbValence.npy' in filename:
    idx = filename.index('emodbValence.npy')
    ar_val.append(filename.pop(idx)) 

if 'emodbArousal.npy' in filename:
    idx = filename.index('emodbArousal.npy')
    ar_val.append(filename.pop(idx)) 

# Big dictionary to store values 
big_dict = {}

# creating paths for labels
for name in ar_val:
    labels_path = os.path.join(inputFolder, name)

    y = np.load(labels_path)

    # Small dictionary to store for each file name
    small_dict = {}

    # creating path names for training(X) data
    for name_x in filename:
        file_x = os.path.join(inputFolder, name_x)
        
        x = np.load(file_x)

        ## preprocessing the trinanig data
        x_scaled = x / 255

        # Set up possible values of parameters to optimize over
        p_grid = {"C": [1,10],
                "gamma": [0.1, 0.01]}

        # We will use a Support Vector Classifier with "rbf" kernel
        svm = SVC(kernel="rbf")
        # array to store scroe
        
        # Loop for each trial
        loopScore = np.zeros(10)

        loop_count = 0
        logo = LeaveOneGroupOut()
    
        # looping through outer group data #10 splits 
        for trainval, test in logo.split(x_scaled, y, group):
            groupval = group[trainval]

            loop_count += 1
            #print "I am runnig %s loop" % loop_count 
        
         ## inner_cv = logo.split(x_scaled[trainval], y[trainval], groups=groupval)
       
            clf = GridSearchCV(estimator= svm, param_grid= p_grid, cv= 3, 
                scoring= 'recall_micro', return_train_score= False)
            clf.fit(x_scaled[trainval], y[trainval])
        
            ##bestScore.append(clf.best_score_)
            ##bestParams.append(clf.best_params_) 
            loopScore[loop_count-1] = clf.score(x_scaled[test], y[test])  
        
        loopMean = loopScore.mean()
        loopStd = loopScore.std()
        # creating a variable for naming in the small dict from the name_x
        small_name_x = name_x[-6:-4]
        small_dict[small_name_x] = (loopMean, loopStd)

    # creating a variable for naming in the big dict from the name
    big_name = name[5:12]

    big_dict[big_name] = small_dict





    stop1 = timeit.default_timer()
    print("***** total excution time for  %f loop = %0.3f minutes"  
        % (loop_count, (stop1 - start) / 60.0))

stop = timeit.default_timer()
print("***** total  programm excution time = %0.3f min" % ((stop - start) / 60.0))



















