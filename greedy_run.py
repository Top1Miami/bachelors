import numpy as np
import os
from utils import read_subsamples
# from build_plots import run_build_plots
from build_model import run_build_model
from algorithm import run_greedy
from build_model import dump_bad_subsamples

directory_name = '3TablesPlots/subsamples'
for file_name in os.listdir(directory_name): # open directory with datasets
    if '.csv' not in file_name: # skip datasets not in csv format
            continue
    with open(directory_name + '/' + file_name, 'r') as fd: # open each file 
        #run_build_plots(x, y) # build all the plots for datasets
        if int(file_name.strip('.csv')) > 5:
            continue
        print("started processing : " + file_name) # logging the start of building procedure
        x, y = read_subsamples(fd) 
        # 4set
        # known_features = np.array([934, 389, 677])
        # good_features = np.array([130, 643, 323, 837, 510, 392, 462, 209, 727, 669, 93, 418, 804, 168, 939, 109, 174, 111, 880, 178, 754, 758, 502, 121, 190, 574, 767])
        # 3set
        # known_features = np.array([65, 337, 339])
        # good_features = np.array([256, 260, 324, 6, 454, 200, 11, 206, 279, 153, 473, 411, 286, 287, 414, 222, 482, 227, 297, 494, 497, 50, 244, 56, 378, 443, 445])
        # 2set
        known_features = np.array([41, 37, 73])
        good_features = np.array([64, 71, 72, 74, 75, 76, 78, 15, 79, 80, 82, 36, 38, 39, 40, 42, 43, 44, 50, 53, 54, 55, 56, 58, 60, 62, 63])
        
        run_greedy(x, y, known_features, good_features) # run and compare proposed algorithm with baseline
