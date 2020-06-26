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
from ITMO_FS.ensembles.measure_based.Melif import Melif
from ITMO_FS.filters.univariate import GLOB_MEASURE, pearson_corr, spearman_corr, information_gain, su_measure, chi2_measure
from ITMO_FS.filters.univariate import UnivariateFilter
from ITMO_FS.filters.univariate import select_k_best, select_best_by_value, select_k_worst
from ITMO_FS.ensembles.measure_based import Melif2Phase
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

def __write_row(html, number_of_test, good_features_train, good_features_test, good_features_mphase, best_point_mphase, rec_mphase, prec_mphase, good_features_m, best_point_m, rec_m, good_features_ontest, best_point_ontest, rec_ontest):
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

    html.write("<td class = \"tableHeader\">")
    html.write("<ul class = \"topList\">")
    for f in sorted(good_features_mphase):
        html.write("<li class = \"topElement\">" + str(f) + "</li>")
    html.write("</ul>")
    html.write("</td>")

    html.write("<td class = \"tableHeader\">")
    html.write("<ul class = \"topList\">")
    for f in best_point_mphase:
        html.write("<li class = \"topElement\">" + str(f) + "</li>")
    html.write("</ul>")
    html.write("</td>")

    html.write("<td class = \"tableHeader\">" + str((rec_mphase, prec_mphase)) + "</td>")

    html.write("<td class = \"tableHeader\">")
    html.write("<ul class = \"topList\">")
    for f in sorted(good_features_m):
        html.write("<li class = \"topElement\">" + str(f) + "</li>")
    html.write("</ul>")
    html.write("</td>")

    html.write("<td class = \"tableHeader\">")
    html.write("<ul class = \"topList\">")
    for f in best_point_m:
        html.write("<li class = \"topElement\">" + str(f) + "</li>")
    html.write("</ul>")
    html.write("</td>")

    html.write("<td class = \"tableHeader\">" + str(rec_m) + "</td>")

    html.write("<td class = \"tableHeader\">")
    html.write("<ul class = \"topList\">")
    for f in sorted(good_features_ontest):
        html.write("<li class = \"topElement\">" + str(f) + "</li>")
    html.write("</ul>")
    html.write("</td>")

    html.write("<td class = \"tableHeader\">")
    html.write("<ul class = \"topList\">")
    for f in best_point_ontest:
        html.write("<li class = \"topElement\">" + str(f) + "</li>")
    html.write("</ul>")
    html.write("</td>")

    html.write("<td class = \"tableHeader\">" + str(rec_ontest) + "</td>") 

    html.write("</tr>")

def __write_post(html):
    html.write("</table>")
    html.write("</body>")
    html.write("</html>")       

starting_list = [0.1, 0.5, 1.0]
starting_dict = {i:starting_list for i in range(0, len(GLOB_MEASURE))}
subsample_size = 70
select_k_number = 6

def loss_two_phase(selected_features, good_features, mapping, to_print=False):
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

def loss_combined(y_pred, y_true, selected_features, good_features, mapping, to_print=False):
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

def methodology_test(html, X, y, good_features, loss_func, number_of_test):
    feature_split = StratifiedKFold(5, shuffle=True)
    feature_marks = feature_mask(X.shape[1], good_features)

    estimator = SVC()
    filters = list(map(lambda measure: UnivariateFilter(measure, select_k_best(30)), GLOB_MEASURE.values()))
    param_grid = starting_dict
    delta = 0.1      

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

            score_train = partial(loss_func, good_features=good_features, mapping=train_mapping)
            score_test_rec = partial(loss_func, good_features=good_features, mapping=test_mapping)
            melif_phase = Melif2Phase(filters, score_train)
            melif_phase.fit(X_ftrain[sample_train], y[sample_train], estimator, select_k_best(select_k_number), X_ftrain[sample_test], y[sample_test], delta=delta, points=ParameterGrid(param_grid))
            melif_phase.run()
            rec_score_phase, prec_score_phase, feat_phase = melif_phase.get_score(X_ftest[sample_train], y[sample_train], X_ftest[sample_test], y[sample_test], score_test_rec)

            un_map_phase = [test_mapping[f] for f in feat_phase]
            good_phase = [f for f in un_map_phase if f in good_features]

            score = f1_score
            melif = Melif(filters, score)
            melif.fit(X_ftrain[sample_train], y[sample_train], estimator, select_k_best(select_k_number), X_ftrain[sample_test], y[sample_test], delta=delta, points=ParameterGrid(param_grid))
            melif.run()
            _, feat_m = melif.get_score(X_ftest[sample_train], y[sample_train], X_ftest[sample_test], y[sample_test])
            un_map_m = [test_mapping[f] for f in feat_m]
            good_m = [f for f in un_map_m if f in good_features]
            score_m = score_test_rec(feat_m)

            score = f1_score
            melif_ontest = Melif(filters, score)
            melif_ontest.fit(X_ftest[sample_train], y[sample_train], estimator, select_k_best(select_k_number), X_ftest[sample_test], y[sample_test], delta=delta, points=ParameterGrid(param_grid))
            feat_ontest = melif_ontest.run()
            un_map_ontest = [test_mapping[f] for f in feat_ontest]
            good_ontest = [f for f in un_map_ontest if f in good_features]
            score_ontest = score_test_rec(feat_ontest)
            
            __write_row(html, number_of_test, good_features_train, good_features_test, good_phase, melif_phase.best_point, rec_score_phase, prec_score_phase, good_m, melif.best_point, score_m, good_ontest, melif_ontest.best_point, score_ontest)

            number_of_test += 1
    return number_of_test

with open('6TablesPlots/shuffled.csv', 'r') as fd: # open each file 
    # run_build_plots(x, y) # build all the plots for datasets
    X, y = read_subsamples(fd)
    # good_features = np.array([64, 193, 194, 453, 455, 458, 203, 463, 336, 338, 24, 281, 153, 344, 472, 347, 475, 415, 35, 169, 105, 493, 378, 433, 50, 241, 442, 443, 318, 319]) # dataset 1 su measure
    # good_features = np.array([128, 576, 386, 643, 899, 195, 712, 521, 15, 849, 274, 854, 664, 793, 345, 414, 287, 868, 486, 745, 621, 622, 239, 114, 819, 374, 248, 570, 251, 764]) # dataset 2 su measure
    # good_features = np.array([576, 577, 324, 521, 266, 267, 268, 269, 522, 459, 460, 402, 211, 598, 351, 352, 549, 296, 431, 239, 240, 241, 379, 181, 374, 377, 378, 571, 572, 318]) # dataset 3 su measure
    # good_features = np.array([64, 195, 197, 70, 74, 76, 83, 212, 85, 216, 26, 28, 158, 162, 163, 228, 229, 102, 232, 106, 237, 50, 178, 52, 252, 183, 185, 124, 254, 191]) # dataset 4 su measure
    # good_features = np.array([256, 321, 66, 517, 326, 138, 398, 784, 403, 916, 83, 534, 851, 380, 40, 169, 681, 937, 744, 362, 46, 876, 560, 561, 945, 117, 376, 315, 60, 190]) # dataset 5 su measure
    good_features = np.array([1, 4, 5, 6, 7, 8, 9, 73, 13, 14, 16, 17, 18, 19, 24, 25, 88, 89, 90, 29, 94, 98, 101, 102, 103, 104, 105, 106, 45, 46]) # dataset 6 su measure
    directory_name = '6TablesPlots/subsamples_melif' # generate the directory for storing results
    subsamples = create_subsamples(directory_name, X, y, subsample_size, 5)
    print('subsamples', len(subsamples))
    # losses = [loss_rec, loss_prec, loss_f1]
    # for loss_func in losses:
    number_of_test = 1    
    html = open('6TablesPlots/features2ph.html', 'w')
    headers = ["number of test", "good feature train", "good features test", "good features selected by MeLiF with 2phase function", 
    "best filter weight vector by MeLiF with 2phase function", "recall score on MeLiF with 2phase function",
    "precision score on MeLiF with 2phase function",
    "good features selected by MeLiF with f1_score as loss function", "best filter weight vector by MeLiF with f1_score as loss function",
    "recall score on MeLiF with f1_score as loss functio", "good features selected by MeLiF trained on test data",
    "best filter weight vector by MeLiF trained on test data", "recall score on MeLiF trained on test data"]
    __write_pre(html, headers)
    for sub_x, sub_y in subsamples:
        print('new subset')
        number_of_test = methodology_test(html, sub_x, sub_y, good_features, loss_two_phase, number_of_test)
    __write_post(html)





