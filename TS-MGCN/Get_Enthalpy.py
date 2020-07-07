import numpy as np
import os
from GetInput_Mol_TS import ExtendConvertToGraph
import random
import sys

def MaxMinNormalization(x, Max, Min):
    x = (x - Min) / (Max - Min)
    return x

def ZscoreNormalization(x, mean, std):
    x = (x - mean) / std
    return x

def Get_Enthalpy(filename):
    nums = len(filename)
    Enthalpy = []
    f = open('./Arkane_output.py', 'r')
    lines = f.readlines()
    for i in range(nums):
        key_word = 'label = \'' + filename[i] + '\''
        # print(key_word)
        iter = 0
        for line in lines:
            iter += 1
            if key_word in line:
                # print(lines[iter])
                Enthalpy_line = lines[iter]
                idx1 = Enthalpy_line.index('(')
                idx2 = Enthalpy_line.index(',')
                Enthalpy.append(float(Enthalpy_line[idx1+1:idx2]))
    f.close()
    Enthalpy = np.asarray(Enthalpy)
    # print(Enthalpy)
    Max = np.max(Enthalpy)
    Min = np.min(Enthalpy)
    for i in range(len(Enthalpy)):
        Enthalpy[i] = MaxMinNormalization(Enthalpy[i], Max, Min)
    print(Max - Min)
    print(Min)
    return Enthalpy

# filePath = '../data/species/'
# filename_raw = os.listdir(filePath)
# filename_raw = np.asarray(filename_raw)
# filename = []
# for i in range(len(filename_raw)):
#     index_dot = filename_raw[i].index('.')
#     filename.append(filename_raw[i][0:index_dot])
# filename = np.asarray(filename)
# random.shuffle(filename)
# file = open('../data/filename_shuffle.txt', 'w+')
# for i in range(len(filename)):
#     file.write(filename[i] + '\n')
# file.close()

file = open('../data/filename_shuffle.txt', 'r')
file_name = file.readlines()
filename = []
for i in range(len(file_name)):
    index_l = file_name[i].index('\n')
    filename.append(file_name[i][0:index_l])
file.close()

features, adj = ExtendConvertToGraph(filename)
Enthalpy = Get_Enthalpy(filename)
print(Enthalpy.shape, features.shape, adj.shape)

np.save('../data/adj.npy', adj)
np.save('../data/features.npy', features)
np.save('../data/Enthalpy.npy', Enthalpy)