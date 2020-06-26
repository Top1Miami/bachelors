import numpy as np
import csv
import os
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedShuffleSplit

def __refactor_labels(y):
    return np.where(y == np.min(y), 0, 1)

def __normalize(x):
    for i in range(x.shape[1]):
        mx = np.max(np.absolute(x[:, i]))
        if mx == 0:
            continue
        for j in range(x.shape[0]):
            x[j][i] /= mx

def feature_mask(feature_number, y):
    mask = np.zeros(feature_number)
    mask[y] = 1
    return mask

def read_and_refactor(fd, directory_name):
    reader = csv.reader(fd) # open reader for csv
    header_csv = reader.__next__() # skipping the data field names
    if header_csv[0] != 'class' and header_csv[0].split(',')[-1].strip("\"") == 'class':
        class_index = -1
    else:
        class_index = np.where(np.array(header_csv) == 'class')[0][0] # get class field number
    data = [] # initilize data holder
    for row in reader: # walk through file and add to data holder
        data.append(list(map(lambda x: float(x), row))) 
    data = np.array(data) # list -> numpy array (#TODO may be use sparse tables instead)
    y = data[:, class_index] # initilize data labels
    y = y.astype(int) # labels -> int
    if np.unique(y).size == 2: # refactor data labels
        y = __refactor_labels(y)
    x = np.delete(data, class_index, 1) # delete y column from data thus creating X sample-feature matrix
    __normalize(x)
    x, y = shuffle(x, y) # shuffle them in case dataset is sorted by y 
    shuffled_fd = open(directory_name + '/shuffled.csv', 'w')
    writer = csv.writer(shuffled_fd)
    for i, _ in enumerate(x):
        writer.writerow(np.append(x[i], y[i]))
    return x, y

def read_subsamples(fd):
    reader = csv.reader(fd)
    data = []
    for row in reader:
        data.append(list(map(lambda x: float(x), row)))
    data = np.array(data)
    y = data[:, data.shape[1] - 1]
    y = y.astype(int)

    x = data[:, :data.shape[1] - 1]
    return x, y

def create_subsamples(directory_name, X, y, subsample_size,  number_of_subsamples=40):
    subsamples = []
    if os.path.exists(directory_name) == False: # check if directory already exists
        print('create')
        os.mkdir(directory_name) # create directory
        print(number_of_subsamples)
        sss = StratifiedShuffleSplit(number_of_subsamples, test_size=subsample_size, random_state=0)
        for i, (train_index, test_index) in enumerate(sss.split(X, y)):
            with open(directory_name + '/' + str(i) + '.csv','w') as fd_csv, open(directory_name + '/' + str(i) + '.txt','w') as fd_txt: # open each file 
                writer = csv.writer(fd_csv)
                for index in test_index:
                    fd_txt.write(str(index) + ' ')
                part_x = X[test_index]
                part_y = y[test_index]
                for i, _ in enumerate(part_x):
                    writer.writerow(np.append(part_x[i], part_y[i]))     
            subsamples.append((X[test_index], y[test_index]))
    else:
        for file_name in os.listdir(directory_name): # open directory with datasets
            if '.csv' not in file_name: # skip datasets not in csv format
                continue
            with open(directory_name + '/' + file_name, 'r') as fd_sub: # open each file
                sub_x, sub_y = read_subsamples(fd_sub)
                subsamples.append((sub_x, sub_y))
    print('subsamples prep' , len(subsamples))
    return subsamples


