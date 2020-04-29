import os
import csv
import numpy as np

def dump_bad_subsamples(x, y, bad_subsamples, directory_name):
    dump_directory_name = directory_name + '/subsamples'
    if os.path.exists(dump_directory_name) == False: # check if directory already exists
        os.mkdir(dump_directory_name) # create directory
    for i, bad_subsample in enumerate(bad_subsamples):
        with open(dump_directory_name + 'i.csv','w') as fd: # open each file 
            number_known, indexes = bad_subsample
            writer = csv.writer(fd)
            part_x = x[indexes]
            part_y = x[indexes]
            for i, _ in enumerate(part_x):
                writer.writerow(np.append(part_x[i], part_y[i]))    
        