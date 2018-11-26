import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import svm
import timeit
from imblearn.over_sampling import SMOTE



start = timeit.default_timer()
X = np.load("/home/mainampati/thesis/X_original_o.npy")
y = np.load("/home/mainampati/thesis/y_berlin_o.npy")
X_resampled, y_resampled = SMOTE().fit_resample(X, y)

no_examples = X.shape[0]
indices = np.arange(no_examples)

no_examples_re = X_resampled.shape[0]
indices_re = np.arange(no_examples_re)


X_scaled = X / 255
X_re_scaled = X_resampled / 255
#print(len(y))

x, x_t, y, y_t, idx1, idx2 = train_test_split(
   X_scaled, y, indices, test_size=0.2, random_state=42)

x_re, x_re_t, y_re, y_re_t, idx1_re, idx2_re = train_test_split(
    X_re_scaled, y_resampled, indices_re, test_size=0.2, random_state=42)


print("number of keyponts considerd...." + str(
    X_scaled.shape[1] / 32))

#loo = LeaveOneOut()
#lin_clf = svm.LinearSVC(C = 0.01)
#lin_clf_re = svm.LinearSVC(C = 0.01)

lin_clf_re = svm.SVC(C = 0.01, kernel = 'rbf', degree= 3,
    decision_function_shape= 'ovr')

lin_clf = svm.SVC(C = 0.01, kernel = 'rbf', degree= 3,
    decision_function_shape= 'ovr')
#scoring = metrics.recall_score(y, average='sample')

scores = cross_val_score(lin_clf, 
    x, y, cv = 10, scoring= 'recall_micro')

scores_re = cross_val_score(lin_clf_re, 
    x_re, y_re, cv = 10, scoring= 'recall_micro')
#print(scores.mean())
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_re.mean(), scores_re.std() * 2))
stop = timeit.default_timer()

print "***** total excution time =  " + str((stop - start) / 60.0)
