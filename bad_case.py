import numpy as np
import os
import csv
from collections import defaultdict
from math import log
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from utils import read_subsamples, create_subsamples
from utils import feature_mask
from ITMO_FS.ensembles.measure_based.MelifLossF import MelifLossF
from ITMO_FS.ensembles.measure_based.Melif2Phase import Melif2Phase
from ITMO_FS.ensembles.measure_based.Melif import Melif
from ITMO_FS.filters.univariate import GLOB_MEASURE, pearson_corr, spearman_corr, information_gain, su_measure, chi2_measure
from ITMO_FS.filters.univariate import UnivariateFilter
from ITMO_FS.filters.univariate import select_k_best, select_best_by_value, select_k_worst
from sklearn.metrics import f1_score, jaccard_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from functools import partial
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import recall_score, precision_score
from sklearn.feature_selection import mutual_info_classif

def __write_pre(html, headers):
    html.write("<!DOCTYPE html>")
    html.write("<head>")
    html.write("<meta charset=\"UTF-8\">")
    html.write("<title>Title</title>")
    html.write("<style>")
    html.write(".tableHeader, .headerElement { padding: 3px; border: 1px solid black; }")
    html.write(".mainTable { border-collapse: collapse; width: 1400px; }")
    html.write(".topElement { list-style-type: none; }")
    html.write("</style>")
    html.write("</head>")
    html.write("<body>")
    html.write("<table class = \"mainTable\">")
    html.write("<tr class=\"tableHeader\">")
    for i in headers:
        html.write("<th class = \"headerElement\">" + i + "</th>")
    html.write("</tr>")

def __write_row(html, number_of_test, good_features_train, good_features_test, goods, best_points, scores):
    html.write("<tr class = \"tableRow\">")
    html.write("<td class = \"tableHeader\">" + str(number_of_test) + "</td>")

    html.write("<td class = \"tableHeader\">")
    html.write("<ul class = \"topList\">")
    for f in sorted(good_features_train):
        html.write("<li class = \"topElement\">" + str(f) + "</li>")
    html.write("</ul>")
    html.write("</td>")

    html.write("<td class = \"tableHeader\">")
    html.write("<ul class = \"topList\">")
    for f in sorted(good_features_test):
        html.write("<li class = \"topElement\">" + str(f) + "</li>")
    html.write("</ul>")
    html.write("</td>")    

    for i in range(len(goods)):
        html.write("<td class = \"tableHeader\">")
        html.write("<ul class = \"topList\">")
        for f in goods[i]:
            html.write("<li class = \"topElement\">" + str(f) + "</li>")
        html.write("</ul>")
        html.write("</td>")

        html.write("<td class = \"tableHeader\">")
        html.write("<ul class = \"topList\">")
        for c in best_points[i]:
            html.write("<li class = \"topElement\">" + str(round(c, 1)) + "</li>")
        html.write("</ul>")
        html.write("</td>")

        html.write("<td class = \"tableHeader\">" + str(round(scores[i][0], 2)) + ', ' + str(round(scores[i][1], 2)) + "</td>")

    html.write("</tr>")

def __write_post(html):
    html.write("</table>")
    html.write("</body>")
    html.write("</html>")       

starting_list = [0.1, 0.5, 1.0]
starting_dict = {i:starting_list for i in range(0, len(GLOB_MEASURE))}
subsample_size = 70
select_k_number = 6

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

def methodology_test(X, y, good_features, other_good, number_of_test):
    feature_split = StratifiedKFold(5, shuffle=True)
    feature_marks = feature_mask(X.shape[1], good_features)

    estimator = SVC()
    filters = list(map(lambda measure: UnivariateFilter(measure, select_k_best(30)), GLOB_MEASURE.values()))
    param_grid = starting_dict
    delta = 0.1      

    for feature_train, feature_test in feature_split.split(X.T, feature_marks):
        
        other_good_train = [f for f in feature_train if f in other_good]
        other_good_test = [f for f in feature_test if f in other_good]
        feature_test_cut = [f for f in feature_test if f not in good_features]

        train_mapping = {i:f for i, f in enumerate(feature_train)}
        test_mapping = {i:f for i, f in enumerate(feature_test_cut)}
        
        sample_split = StratifiedKFold(5)
        
        for sample_train, sample_test in sample_split.split(X, y):
            print('new test number:', number_of_test)
            
            X_ftrain = X[:, feature_train]
            X_ftest = X[:, feature_test_cut]
            good_features_test = [value for value in test_mapping.values() if value in good_features]
            good_features_train = [value for value in train_mapping.values() if value in good_features]

            score_train_prec = partial(loss_prec, good_features=good_features, mapping=train_mapping)
            score_test_prec = partial(loss_prec, good_features=good_features, mapping=test_mapping)
            
            melif_prec = MelifLossF(filters, score_train_prec)
            melif_prec.fit(X_ftrain[sample_train], y[sample_train], select_k_best(24), delta=delta, points=ParameterGrid(param_grid))
            melif_prec.run()
            feat_prec = melif_prec.transform(X_ftest[sample_train], y[sample_train], select_k_best(6))
            sel_prec = [test_mapping[f] for f in feat_prec]
            good_prec = [test_mapping[f] for f in feat_prec if test_mapping[f] in good_features]

            score_test_2phase = partial(loss_2phase, good_features=other_good_test, mapping=test_mapping)
            print('selected features', sorted(sel_prec))
            print('other from top', sorted(other_good_test))
            print('recall and precision', score_test_2phase(feat_prec))
            
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
    with open(str(number) + 'TablesPlots/shuffled.csv', 'r') as fd: # open each file 
        X, y = read_subsamples(fd)

        univ_filter = UnivariateFilter(su_measure, select_k_best(50))
        univ_filter.fit(X, y)

        other_good = set(univ_filter.selected_features).difference(set(good_features[number - 1]))
        print('other than good', other_good)
        directory_name = str(number) + 'TablesPlots/subsamples_melif' # generate the directory for storing results
        subsamples = create_subsamples(directory_name, X, y, subsample_size, 5)
        print('subsamples', len(subsamples))
        number_of_test = 1
        for sub_x, sub_y in subsamples:
            print('new subset')
            number_of_test = methodology_test(sub_x, sub_y, good_features[number - 1], other_good, number_of_test)



