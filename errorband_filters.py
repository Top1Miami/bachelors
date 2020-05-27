import numpy as np
import os
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from math import log
from functools import partial
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import recall_score, precision_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.svm import SVC
from sklearn.metrics import f1_score, jaccard_score
from ITMO_FS.ensembles.measure_based.MelifLossF import MelifLossF
from ITMO_FS.ensembles.measure_based.Melif2Phase import Melif2Phase
from ITMO_FS.ensembles.measure_based.Melif import Melif
from ITMO_FS.filters.univariate import GLOB_MEASURE, pearson_corr, spearman_corr, information_gain, su_measure
from ITMO_FS.filters.univariate import UnivariateFilter
from ITMO_FS.filters.univariate import select_k_best, select_best_by_value, select_k_worst
from ITMO_FS.wrappers.deterministic import BackwardSelection
from ITMO_FS.wrappers.deterministic import SequentialForwardSelection
from utils import read_subsamples, create_subsamples
from utils import feature_mask

def loss_2phase(selected_features, good_features, mapping, to_print=False):
    f_true = np.zeros(len(mapping))
    g_feat = []
    for key, value in mapping.items():
        if value in good_features:
            f_true[key] = 1
            g_feat.append(value)
    f_pred = np.zeros(len(mapping))
    sel_feat = []
    for f in selected_features:
        f_pred[f] = 1
        sel_feat.append(mapping[f])
    rec_score = recall_score(f_true, f_pred)
    prec_score = precision_score(f_true, f_pred)
    return rec_score, prec_score

def loss_combined(selected_features, good_features, mapping, to_print=False):
    f1_marks = f1_score(y_pred, y_true)
    f_true = np.zeros(len(mapping))
    for key, value in mapping.items():
        if value in good_features:
            f_true[key] = 1
    f_pred = np.zeros(len(mapping))
    sel_feat = []
    for f in selected_features:
        f_pred[f] = 1
        sel_feat.append(mapping[f])
    f1_features = f1_score(f_true, f_pred)
    return f1_marks + f1_features

def common_loss(selected_features, good_features, mapping, score_function, to_print=False):
    f_true = np.zeros(len(mapping))
    g_feat = []
    for key, value in mapping.items():
        if value in good_features:
            f_true[key] = 1
            g_feat.append(value)
    f_pred = np.zeros(len(mapping))
    sel_feat = []
    for f in selected_features:
        f_pred[f] = 1
        sel_feat.append(mapping[f])
    score = score_function(f_true, f_pred)
    return score

def loss_rec(selected_features, good_features, mapping, to_print=False):
    return common_loss(selected_features, good_features, mapping, recall_score, to_print=to_print)

def loss_prec(selected_features, good_features, mapping, to_print=False):
    return common_loss(selected_features, good_features, mapping,  precision_score, to_print=to_print)

def loss_jaccard(selected_features, good_features, mapping, to_print=False):
    return common_loss(selected_features, good_features, mapping, jaccard_score, to_print=to_print)

def loss_f1(selected_features, good_features, mapping, to_print=False):
    return common_loss(selected_features, good_features, mapping, f1_score, to_print=to_print)       

def methodology_test(X, y, good_features, number_of_test, scores_filters, k):
    feature_split = StratifiedKFold(5, shuffle=True)
    feature_marks = feature_mask(X.shape[1], good_features)

    filters = list(map(lambda measure: UnivariateFilter(measure, select_k_best(k)), GLOB_MEASURE.values()))
    
    for feature_train, feature_test in feature_split.split(X.T, feature_marks):
        
        train_mapping = {i:f for i, f in enumerate(feature_train)}
        test_mapping = {i:f for i, f in enumerate(feature_test)}
        
        sample_split = StratifiedKFold(5)
        
        for sample_train, sample_test in sample_split.split(X, y):
            print('new test number:', number_of_test)
            
            X_ftrain = X[:, feature_train]
            X_ftest = X[:, feature_test]
            good_features_test = [value for value in test_mapping.values() if value in good_features]
            good_features_train = [value for value in train_mapping.values() if value in good_features]

            score_test_2phase = partial(loss_2phase, good_features=good_features, mapping=test_mapping)
            
            for filter_ in filters:
                filter_.fit(X_ftest[sample_train], y[sample_train])
                sel_feat = filter_.selected_features
                rec, prec = score_test_2phase(sel_feat)
                scores_filters.append([rec, prec, k, filter_.measure.__name__])

            number_of_test += 1
    return number_of_test

for number in range(1, 7):
    good_features = np.array([[64, 193, 194, 453, 455, 458, 203, 463, 336, 338, 24, 281, 153, 344, 472, 347, 475, 415, 35, 169, 105, 493, 378, 433, 50, 241, 442, 443, 318, 319], # dataset 1 su measure
    [128, 576, 386, 643, 899, 195, 712, 521, 15, 849, 274, 854, 664, 793, 345, 414, 287, 868, 486, 745, 621, 622, 239, 114, 819, 374, 248, 570, 251, 764], # dataset 2 su measure
    [576, 577, 324, 521, 266, 267, 268, 269, 522, 459, 460, 402, 211, 598, 351, 352, 549, 296, 431, 239, 240, 241, 379, 181, 374, 377, 378, 571, 572, 318], # dataset 3 su measure
    [64, 195, 197, 70, 74, 76, 83, 212, 85, 216, 26, 28, 158, 162, 163, 228, 229, 102, 232, 106, 237, 50, 178, 52, 252, 183, 185, 124, 254, 191], # dataset 4 su measure
    [256, 321, 66, 517, 326, 138, 398, 784, 403, 916, 83, 534, 851, 380, 40, 169, 681, 937, 744, 362, 46, 876, 560, 561, 945, 117, 376, 315, 60, 190], # dataset 5 su measure
    [1, 4, 5, 6, 8, 9, 73, 13, 14, 16, 17, 18, 19, 24, 25, 88, 89, 90, 29, 94, 98, 101, 102, 103, 104, 105, 106, 45, 46]]) # dataset 6 su measure
    subsample_size = 70
    select_k_number = list(range(1, 7)) 
    with open(str(number) + 'TablesPlots/shuffled.csv', 'r') as fd: # open each file 
        sns.set(style="darkgrid")
        X, y = read_subsamples(fd)
        directory_name = str(number) + 'TablesPlots/subsamples_melif' # generate the directory for storing results
        subsamples = create_subsamples(directory_name, X, y, subsample_size, 5)
        print('subsamples', len(subsamples))
        number_of_test = 1
        scores_filters = []
        for sub_x, sub_y in subsamples:
            print('new subset')
            for k in select_k_number:
                number_of_test = methodology_test(sub_x, sub_y, good_features[number - 1], number_of_test, scores_filters, k)
        
        dump_points = open(str(number) + 'TablesPlots/dump_points_filter.txt', 'w')
        dump_points.write(str(len(scores_filters)) + ' ' + str(len(scores_filters[0])))
        for i in range(len(scores_filters)):
            for j in range(len(scores_filters[i])):
                dump_points.write(str(scores_filters[i][j]) + ' ')
            dump_points.write('\n')
        
        df1 = pd.DataFrame(data=scores_filters, index=range(len(scores_filters)), columns=["recall", "precision", "cutting_rule", "loss"])
        sns.lineplot(x="cutting_rule", y="precision",
                 hue="loss",
                 data=df1)
        plt.savefig(str(number) + 'TablesPlots/scores_prec_filter.png')
        plt.close()
        
        df2 = pd.DataFrame(data=scores_filters, index=range(len(scores_filters)), columns=["recall", "precision", "cutting_rule", "loss"])
        sns.lineplot(x="cutting_rule", y="recall",
                 hue="loss",
                 data=df2)
        plt.savefig(str(number) + 'TablesPlots/scores_rec_filter.png')
        plt.close()



