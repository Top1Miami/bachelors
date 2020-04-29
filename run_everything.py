import numpy as np
import os
from utils import read_and_refactor
# from build_plots import run_build_plots
from build_model import run_build_model
from algorithm import run_and_compare_fs
from build_model import dump_bad_subsamples

for file_name in os.listdir("datasets"): # open directory with datasets
    with open('datasets/' + file_name, 'r') as fd: # open each file 
        if '.csv' not in file_name: # skip datasets not in csv format
            continue
        if int(file_name[0]) != 4: # test only on dataset number 4
            continue
        print("started processing : " + file_name) # logging the start of building procedure
        directory_name = file_name[:-4] + 'TablesPlots' # generate the directory for storing results 
        if os.path.exists(directory_name) == False: # check if directory already exists
            os.mkdir(directory_name) # create directory
        x, y = read_and_refactor(fd, directory_name) # read and refactor input data
        #run_build_plots(x, y) # build all the plots for datasets
        bad_subsamples, known_features, good_features = run_build_model(x, y, directory_name) # build model for feature selection testing
        print('known:', known_features)
        print('good:', good_features)
        dump_bad_subsamples(x, y, bad_subsamples, directory_name)
        # run_and_compare_fs(x, y, bad_subsamples, known_features, good_features, directory_name) # run and compare proposed algorithm with baseline