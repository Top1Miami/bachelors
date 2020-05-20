from UnivariateFilter import UnivariateFilter
import measures
import csv
import numpy as np

with open('datasets/4.csv', 'r') as fd: # open each file 
    reader = csv.reader(fd) # open reader for csv
    header_csv = reader.__next__() # skipping the data field names
    class_index = np.where(np.array(header_csv) == 'class')[0][0] # get class field number
    data = [] # initilize data holder
    for row in reader: # walk through file and add to data holder
        data.append(list(map(lambda x: float(x), row))) 
    data = np.array(data) # list -> numpy array (#TODO may be use sparse tables instead)
    y = data[:, class_index] # initilize data labels
    y = y.astype(int) # labels -> int
    
    # if np.unique(y).size == 2: # refactor data labels
    #     y = __refactor_labels(y)
    x = np.delete(data, class_index, 1) # delete y column from data thus creating X sample-feature matrix
    univ_filter = UnivariateFilter(measures.pearson_corr, measures.select_k_best(30))
    univ_filter.fit(x, y)
    print(univ_filter.selected_features)
    