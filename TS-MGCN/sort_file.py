import numpy as np

def sort_file(All, modelName, iter):
    np.set_printoptions(precision=4)
    file = open('../data/filename_shuffle.txt', 'r')
    filename = file.readlines()
    file.close()
    back_filename = np.array(filename)
    cross_iterNum = int(len(filename)/10)
    back_filename[iter*cross_iterNum:(iter+1)*cross_iterNum] = filename[8*cross_iterNum:9*cross_iterNum]
    back_filename[8*cross_iterNum:9*cross_iterNum] = filename[iter*cross_iterNum:(iter+1)*cross_iterNum]
    All = np.asarray(All)
    sort_file = np.c_[All, back_filename]
    sort_file = sort_file[sort_file[:, 2].argsort()]
    f = open('./sort_file/' + modelName + '_sort.txt', 'w+')
    for i in range(len(sort_file)):
        f.write(str(sort_file[i, 0]) + '\t' + str(sort_file[i, 1]) + '\t' + str(sort_file[i, 2]) + '\t' + str(sort_file[i, 3]))
    f.close()

def sort_filefinal(All, modelName):
    np.set_printoptions(precision=4)
    file = open('../data/filename_shuffle.txt', 'r')
    filename = file.readlines()
    file.close()
    filename = np.array(filename)
    All = np.asarray(All)
    sort_file = np.c_[All, filename]
    sort_file = sort_file[sort_file[:, 2].argsort()]
    f = open('./sort_file/' + modelName + '_sort.txt', 'w+')
    for i in range(len(sort_file)):
        f.write(str(sort_file[i, 0]) + '\t' + str(sort_file[i, 1]) + '\t' + str(sort_file[i, 2]) + '\t' + str(sort_file[i, 3]))
    f.close()

def sort_only(modelName):
    file = np.loadtxt('./txt/' + modelName + '_Test_file.txt')
    file = file[file[:, 2].argsort()]
    f = open('./sort_only/' + modelName + '_sort.txt', 'w+')
    for i in range(len(file)):
        f.write(str(file[i, 0]) + '\t' + str(file[i, 1]) + '\t' + str(file[i, 2]) + '\n')
    f.close()