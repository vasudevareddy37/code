import os
import shutil
import random


sourcedir = '/home/mainampati/databases/spctro_db/Emo_db'
destidir = '/home/mainampati/databases/spctro_db/Emo_cdb'

files = [file for file in os.listdir(sourcedir)]
# if os.path.isfile(os.path.join(sourcedir, file))

random_percen = 0.7
idx = []

random.seed(1)
# Number of random files
no_random_files = int(len(files) * random_percen)
# Selecting the files randomly 
random_files = random.sample(files, no_random_files) 

# Copy or moving the files 
for x in random_files:
    idx.append(files.index(x)) # Index of selected files in orginal folder
    shutil.copy(os.path.join(sourcedir, x), destidir)

# Cross verification by examining the destination folder
files2 = [file for file in os.listdir(destidir)]
# if os.path.isfile(os.path.join(destidir, file))

print len(files2) == len(idx)
idx2 = []
for k in files2:
    idx2.append(files.index(k))

#idx.sort()