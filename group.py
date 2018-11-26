import os
import numpy as np

# import timeit


inputFolder = '/home/mainampati/thesis/Emo_db'
suffix = '.png'

filename  = os.listdir(inputFolder)
L = len(filename)
group = np.zeros((L,))
print"shape of the group..." + str(group.shape) 

for i in range (0, L):

    base_filename = filename[i]
    #print(base_filename)
    #name = os.path.join(inputFolder, base_filename)
    #print(base_filename[0:2])
    if base_filename[0:2] == '03':
        #group = np.r_[group, 2]
        group[i] = 1
    
    elif base_filename[0:2] == '08':
        #group = np.r_[group, 2]
        group[i] = 2
    elif base_filename[0:2] == '09':
        #group = np.r_[group, 3]
        group[i] = 3
    elif base_filename[0:2] == '10':
        #group = np.r_[group, 4]
        group[i] = 4
    elif base_filename[0:2] == '11':
        #group = np.r_[group, 0:2]
        group[i] = 5
    elif base_filename[0:2] == '12':
        #group = np.r_[group, 6]
        group[i] = 6
    elif base_filename[0:2] == '13':
        #group = np.r_[group, 6]
        group[i] = 7
    elif base_filename[0:2] == '14':
        #group = np.r_[group, 6]
        group[i] = 8
    elif base_filename[0:2] == '15':
        #group = np.r_[group, 6]
        group[i] = 9
    elif base_filename[0:2] == '16':
        #group = np.r_[group, 6]
        group[i] = 10
    else:
        #group = np.r_[group, 7]
        print "problem with the file %s " % base_filename

g = np.copy(group)

np.save('/home/mainampati/thesis/group.npy', g)
