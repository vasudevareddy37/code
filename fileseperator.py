import os, errno
import shutil
import random


sourcedir = '/home/mainampati/thesis/eeee/Emo_db'
destidir = '/home/mainampati/thesis/eeee/Emo_cdb'


try:
    os.makedirs(destidir)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

## puting the file names in the list from the source dir
files = [file for file in os.listdir(sourcedir)]

# if os.path.isfile(os.path.join(sourcedir, file))
## length of the trainng set
testSet_percen = 0.7

idx = []

random.seed(1)
# Number of random files
no_test_files = int(len(files) * testSet_percen)

# Selecting the files randomly 
random_files = random.sample(files, no_test_files) 

# Copy or moving the files 
for x in random_files:
    idx.append(files.index(x)) # Index of selected files in orginal folder
    shutil.move(os.path.join(sourcedir, x), destidir)

# Cross verification by examining the destination folder
files2 = [file for file in os.listdir(destidir)]
# if os.path.isfile(os.path.join(destidir, file))

print len(files2) == len(idx)

# idx2 = []
# for k in files2:
#     idx2.append(files.index(k))

#idx.sort()