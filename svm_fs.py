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
from sklearn.metrics import f1_score, jaccard_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from functools import partial
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import recall_score, precision_score
from sklearn.feature_selection import mutual_info_classif
from algorithm import SemiSupFS

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

def __write_row(html, number_of_test, good_features_train, good_features_test, kernel, features_semi, score_semi):
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

    html.write("<td class = \"tableHeader\">" + kernel + "</td>")

    html.write("<td class = \"tableHeader\">")
    html.write("<ul class = \"topList\">")
    for f in sorted(features_semi):
        html.write("<li class = \"topElement\">" + str(f) + "</li>")
    html.write("</ul>")
    html.write("</td>")
    
    html.write("<td class = \"tableHeader\">" + str(score_semi) + "</td>")

    html.write("</tr>")

def __write_post(html):
    html.write("</table>")
    html.write("</body>")
    html.write("</html>")       

starting_list = [0.1, 0.5, 1.0]
starting_dict = {i:starting_list for i in range(0, len(GLOB_MEASURE))}
select_k_number = 6

def loss_2phase(y_pred, y_true, selected_features, good_features, mapping, to_print=False):
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

def loss_no_mapping(selected_features, good_features, number_of_features, func):
    f_true = np.zeros(number_of_features)
    f_pred = np.zeros(number_of_features)
    for f in good_features:
        f_true[f] = 1
    for f in selected_features:
        f_pred[f] = 1
    return func(f_true, f_pred)

def methodology_test(html, X, y, good_features, number_of_test, kernel_parameters):
    feature_split = StratifiedKFold(5, shuffle=True)
    feature_marks = feature_mask(X.shape[1], good_features)
    best_prec = 0.0
    best_sol = []    
    for feature_train, feature_test in feature_split.split(X.T, feature_marks):
        train_mapping = {i:f for i, f in enumerate(feature_train)}
        test_mapping = {i:f for i, f in enumerate(feature_test)}
        
        for kernel in kernel_parameters:
            print('new test number:', number_of_test)
            print(kernel)
            
            X_ftrain = X[:, feature_train]
            X_ftest = X[:, feature_test]

            good_features_test = [value for value in test_mapping.values() if value in good_features]
            good_features_train = [value for value in train_mapping.values() if value in good_features]

            fs_alg = SemiSupFS(kernel)
            fs_alg.run(X, (good_features_train))
            features_semi = fs_alg.selected_features

            score_semi = loss_no_mapping(features_semi, good_features_test, X.shape[1], recall_score), loss_no_mapping(features_semi, good_features_test, X.shape[1], precision_score)
            
            if score_semi[1] > 0.5:
                __write_row(html, number_of_test, good_features_train, good_features_test, kernel, features_semi, score_semi)
            number_of_test += 1
            if score_semi[1] > best_prec:
                best_prec = score_semi[1]
                best_sol = number_of_test, good_features_train, good_features_test, kernel, features_semi, score_semi
    print(best_sol)

    return number_of_test

with open('4TablesPlots/shuffled.csv', 'r') as fd: # open each file 
    # run_build_plots(x, y) # build all the plots for datasets
    
    kernel_parameters = []

    for degree in range(1, 5):
        for coef in range(0, 10):
            coef0 = float(coef) / 10     
            for nu in range(1, 100):
                nu0 = float(nu) / 100
                kernel_parameters.append('poly ' + str(degree) + ' ' + str(coef0) + ' ' + str(nu0) + ' scale')
                kernel_parameters.append('poly ' + str(degree) + ' ' + str(coef0) + ' ' + str(nu0) + ' auto')
        for coef in range(2, 10):
            for nu in range(1, 100):
                nu0 = float(nu) / 100
                kernel_parameters.append('poly ' + str(degree) + ' ' + str(coef) + ' ' + str(nu0) + ' scale')
                kernel_parameters.append('poly ' + str(degree) + ' ' + str(coef) + ' ' + str(nu0) + ' auto')

    for coef in range(0, 10):
        coef0 = float(coef) / 10   
        for nu in range(1, 100):
            nu0 = float(nu) / 100
            kernel_parameters.append('sigmoid ' + str(coef0) + ' ' + str(nu0) + ' scale')
            kernel_parameters.append('sigmoid ' + str(coef0) + ' ' + str(nu0) + ' auto')
    for coef in range(2, 10):
        for nu in range(1, 100):
            nu0 = float(nu) / 100
            kernel_parameters.append('sigmoid ' + str(coef) + ' ' + str(nu0) + ' scale')
            kernel_parameters.append('sigmoid ' + str(coef) + ' ' + str(nu0) + ' auto')
    for nu in range(1, 100):
        nu0 = float(nu) / 100
        kernel_parameters.append('rbf ' + str(nu0) + ' scale')
        kernel_parameters.append('rbf ' + str(nu0) + ' auto')

    for nu in range(1, 100):
        nu0 = float(nu) / 100
        kernel_parameters.append('linear ' + str(nu0))

    X, y = read_subsamples(fd)
    # good_features = np.array([64, 193, 194, 453, 455, 458, 203, 463, 336, 338, 24, 281, 153, 344, 472, 347, 475, 415, 35, 169, 105, 493, 378, 433, 50, 241, 442, 443, 318, 319]) # dataset 1 su measure
    # good_features = np.array([128, 576, 386, 643, 899, 195, 712, 521, 15, 849, 274, 854, 664, 793, 345, 414, 287, 868, 486, 745, 621, 622, 239, 114, 819, 374, 248, 570, 251, 764]) # dataset 2 su measure
    # good_features = np.array([576, 577, 324, 521, 266, 267, 268, 269, 522, 459, 460, 402, 211, 598, 351, 352, 549, 296, 431, 239, 240, 241, 379, 181, 374, 377, 378, 571, 572, 318]) # dataset 3 su measure
    good_features = np.array([64, 195, 197, 70, 74, 76, 83, 212, 85, 216, 26, 28, 158, 162, 163, 228, 229, 102, 232, 106, 237, 50, 178, 52, 252, 183, 185, 124, 254, 191]) # dataset 4 su measure
    # good_features = np.array([256, 321, 66, 517, 326, 138, 398, 784, 403, 916, 83, 534, 851, 380, 40, 169, 681, 937, 744, 362, 46, 876, 560, 561, 945, 117, 376, 315, 60, 190]) # dataset 5 su measure
    # good_features = np.array([1, 4, 5, 6, 7, 8, 9, 73, 13, 14, 16, 17, 18, 19, 24, 25, 88, 89, 90, 29, 94, 98, 101, 102, 103, 104, 105, 106, 45, 46]) # dataset 6 su measure
    directory_name = '4TablesPlots/subsamples_svm' # generate the directory for storing results
    subsample_size = 100
    subsamples = create_subsamples(directory_name, X, y, subsample_size, 1)
    print('subsamples', len(subsamples))
    number_of_test = 1    
    html = open('4TablesPlots/features_svm.html', 'w')
    headers = ["number of test", "good feature train", "good features test", "kernel",
    "good features selected with one-class SVM", "recall, precision score on one-class SVM",]
    __write_pre(html, headers)
    for sub_x, sub_y in subsamples:
        print('new subset')
        print(len(kernel_parameters))
        number_of_test = methodology_test(html, sub_x, sub_y, good_features, number_of_test, kernel_parameters)
    __write_post(html)

# poly 1 0.0 0.67 auto
# poly 1 0.0 0.83 auto
# poly 1 0.0 0.94 scale


