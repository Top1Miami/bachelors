import numpy as np
import csv
from sklearn.utils import shuffle

def __refactor_labels(y):
    return np.where(y == np.min(y), 0, 1)

def read_and_refactor(fd):
    reader = csv.reader(fd) # open reader for csv
    header_csv = reader.__next__() # skipping the data field names
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
    x, y = shuffle(x, y) # shuffle them in case dataset is sorted by y 
    return x, y