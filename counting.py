import numpy as np
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt


#group = np.load("/home/mainampati/thesis/group.npy") 
y = np.load("/home/mainampati/thesis/y_berlin_o.npy")
x = np.load("/home/mainampati/thesis/X_original_o.npy")


X_resampled, y_resampled = SMOTE().fit_resample(x, y)

print(sorted(Counter(y_resampled).items()))
print len(y_resampled)
print(sorted(Counter(y).items()))
print len(y)

# Initialinzing the counter
cnt  = Counter()

# making dict by counting according to labels
for element in y_resampled:
    cnt[element] += 1

# seperating the keys and values 
labels = cnt.keys()
#print labels
example = cnt.values()

plt.bar(labels, example, color=['black', 'red', 'green', 'blue', 'cyan', 'yellow', 'magenta'])


plt.xlabel('Classes')
plt.ylabel('Frequency')
plt.title("Data Distribution")
plt.legend()
plt.show()