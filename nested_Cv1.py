import timeit
#import collections
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import  recall_score
from sklearn.svm import SVC


## Starting timer
start = timeit.default_timer()

NUM_TRIALS = 1
## Loading the  dataset
x = np.load("/home/mainampati/thesis/emo_db/spectro_db3Re/spectro_db3E.npy")
y = np.load("/home/mainampati/thesis/emo_db/spectro_db3Re/emodbArousal.npy")
group = np.load("/home/mainampati/thesis/emo_db/spectro_db3Re/gruop.npy") 

# analysing gruop data
## unique, counts = np.unique(group, return_counts=True)
## print(dict(zip(unique, counts)))
## print collections.Counter(group)

## preprocessing the trinanig datanes
x_scaled = x / 255

# Set up possible values of parameters to optimize over
p_grid = {"C": [0.1, 1, 10, 100],
          "gamma": [1, 0.01, 0.001]}

# We will use a Support Vector Classifier with "rbf" kernel
svm = SVC(kernel="rbf")

# array to store scroe
nested_scores = np.zeros(NUM_TRIALS)
bestParams = []
bestScore = []
loopScore = np.zeros(10)

# Loop for each trial
for i in range(NUM_TRIALS):

    # Choose cross-validation techniques for the inner and outer loops,
    # independently of the dataset.
    # E.g "GroupKFold", "LeaveOneOut", "LeaveOneGroupOut", etc.
    ##inner_cv = KFold(n_splits=4, shuffle=True, random_state=i)
    loop_count = 0
    logo = LeaveOneGroupOut()
    
    ##print "number of outer splits %s" % logo.get_n_splits(groups = group)
    
    # looping through outer group data #10 splits 
    for trainval, test in logo.split(x_scaled, y, group):
        groupval = group[trainval]

        loop_count += 1
        print "I am runnig %s loop" % loop_count 
        
        # looping through inner gruop data # 9 splits
        ## for train, test in logo.split(x_scaled[trainval], y[trainval], groupval)
        ## print "number of inner splits %s" % logo.get_n_splits(groups = groupval)
        ## inner_group = groupval[train]
        #inner_cv = logo.split(x_scaled[trainval], y[trainval], groups=groupval)
       
        clf = GridSearchCV(estimator= svm, param_grid= p_grid, cv= 3, 
            scoring= 'recall', return_train_score= False)
        clf.fit(x_scaled[trainval], y[trainval])
        
        bestScore.append(clf.best_score_)
        bestParams.append(clf.best_params_) 
        loopScore[loop_count-1] = clf.score(x_scaled[test], y[test])   

        stop1 = timeit.default_timer()
        print("***** total excution time for  %f loop = %0.3f minutes"  
            % (loop_count, (stop1 - start) / 60.0))

stop = timeit.default_timer()
print("***** total  programm excution time = %0.3f min" % ((stop - start) / 60.0))

print "final average score on database is"
finalScore = loopScore.mean() * 100
std = loopScore.std() * 100
print ("final mean score is: %s and standard deviation is %s" % (finalScore, std))





















