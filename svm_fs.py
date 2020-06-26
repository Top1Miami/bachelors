import numpy as np
import os
import csv
import random
from collections import defaultdict
from statistics import mean 
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
import matplotlib.pyplot as plt


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
    
    html.write("<td class = \"tableHeader\">" + str(round(score_semi[0], 2)) + ', ' + str(round(score_semi[1], 2)) + "</td>")

    html.write("</tr>")

def __write_post(html):
    html.write("</table>")
    html.write("</body>")
    html.write("</html>")        

def loss_no_mapping(selected_features, good_features, number_of_features, func):
    f_true = np.zeros(number_of_features)
    f_pred = np.zeros(number_of_features)
    for f in good_features:
        f_true[f] = 1
    for f in selected_features:
        f_pred[f] = 1
    return func(f_true, f_pred)


def methodology_test(html, X, y, good_features, number_of_test, kernel_parameters, scores):
    feature_split = StratifiedKFold(6, shuffle=True)
    feature_marks = feature_mask(X.shape[1], good_features)
    for feature_tv, feature_test in feature_split.split(X.T, feature_marks):
        tv_mapping = {i:f for i, f in enumerate(feature_tv)}
        test_mapping = {i:f for i, f in enumerate(feature_test)}
        
        good_tv = [k for k, v in tv_mapping.items() if v in good_features]
        
        good_train = random.sample(good_tv, 20)
        good_validate = list(set(good_tv).difference(set(good_train)))
        best_score_prec = 0.0
        best_score_rec = 0.0
        best_kernel = ''
        for kernel in kernel_parameters:
            fs_alg = SemiSupFS(kernel)
            fs_alg.run(X[:, feature_tv], good_train)
            features_semi = fs_alg.selected_features

            # orig_true = [tv_mapping[f] for f in good_validate]
            # orig_sel = [tv_mapping[f] for f in features_semi]
            rec, prec = loss_no_mapping(features_semi, good_validate, len(feature_tv), recall_score), loss_no_mapping(features_semi, good_validate, len(feature_tv), precision_score)
            if best_score_prec < prec:
                best_score_prec = prec
                best_score_rec = rec
                best_kernel = kernel
            elif best_score_prec == prec and best_score_rec < rec:
                best_score_rec = rec
                best_kernel = kernel
        if best_kernel == '':
            continue
        good_tv_v = [f for f in feature_tv if f in good_features]
        fs_alg = SemiSupFS(best_kernel)
        fs_alg.run(X, good_tv_v) 
        features_semi = fs_alg.selected_features
        good_test = [f for f in feature_test if f in good_features]
        score_test = loss_no_mapping(features_semi, good_test, X.shape[1], recall_score), loss_no_mapping(features_semi, good_test, X.shape[1], precision_score)        
        
        # no_test_list = list(set(range(0, X.shape[1])).difference(set(good_test)))

        __write_row(html, number_of_test, good_tv_v, good_test, best_kernel, features_semi, score_test)

        scores[len(features_semi)].append(score_test)

        number_of_test += 1
    return number_of_test

for number in range(1, 7):
    with open(str(number) + 'TablesPlots/shuffled.csv', 'r') as fd: # open each file 
        # run_build_plots(x, y) # build all the plots for datasets
        
        kernel_parameters = []

        for degree in range(2, 5):
            for coef in range(0, 10):
                coef0 = float(coef) / 10     
                for nu in range(80, 100):
                    nu0 = float(nu) / 100
                    kernel_parameters.append('poly ' + str(degree) + ' ' + str(coef0) + ' ' + str(nu0) + ' scale')
                    kernel_parameters.append('poly ' + str(degree) + ' ' + str(coef0) + ' ' + str(nu0) + ' auto')
            for coef in range(2, 10):
                for nu in range(80, 100):
                    nu0 = float(nu) / 100
                    kernel_parameters.append('poly ' + str(degree) + ' ' + str(coef) + ' ' + str(nu0) + ' scale')
                    kernel_parameters.append('poly ' + str(degree) + ' ' + str(coef) + ' ' + str(nu0) + ' auto')

        for coef in range(0, 10):
            coef0 = float(coef) / 10   
            for nu in range(80, 100):
                nu0 = float(nu) / 100
                kernel_parameters.append('sigmoid ' + str(coef0) + ' ' + str(nu0) + ' scale')
                kernel_parameters.append('sigmoid ' + str(coef0) + ' ' + str(nu0) + ' auto')
        
        for coef in range(2, 10):
            for nu in range(80, 100):
                nu0 = float(nu) / 100
                kernel_parameters.append('sigmoid ' + str(coef) + ' ' + str(nu0) + ' scale')
                kernel_parameters.append('sigmoid ' + str(coef) + ' ' + str(nu0) + ' auto')

        for nu in range(80, 100):
            nu0 = float(nu) / 100
            kernel_parameters.append('rbf ' + str(nu0) + ' scale')
            kernel_parameters.append('rbf ' + str(nu0) + ' auto')
            kernel_parameters.append('linear ' + str(nu0))

        scores = defaultdict(list)

        X, y = read_subsamples(fd)
        good_features = np.array([[64, 193, 194, 453, 455, 458, 203, 463, 336, 338, 24, 281, 153, 344, 472, 347, 475, 415, 35, 169, 105, 493, 378, 433, 50, 241, 442, 443, 318, 319], # dataset 1 su measure
        [128, 576, 386, 643, 899, 195, 712, 521, 15, 849, 274, 854, 664, 793, 345, 414, 287, 868, 486, 745, 621, 622, 239, 114, 819, 374, 248, 570, 251, 764], # dataset 2 su measure
        [576, 577, 324, 521, 266, 267, 268, 269, 522, 459, 460, 402, 211, 598, 351, 352, 549, 296, 431, 239, 240, 241, 379, 181, 374, 377, 378, 571, 572, 318], # dataset 3 su measure
        [64, 195, 197, 70, 74, 76, 83, 212, 85, 216, 26, 28, 158, 162, 163, 228, 229, 102, 232, 106, 237, 50, 178, 52, 252, 183, 185, 124, 254, 191], # dataset 4 su measure
        [256, 321, 66, 517, 326, 138, 398, 784, 403, 916, 83, 534, 851, 380, 40, 169, 681, 937, 744, 362, 46, 876, 560, 561, 945, 117, 376, 315, 60, 190], # dataset 5 su measure
        [1, 4, 5, 6, 7, 8, 9, 73, 13, 14, 16, 17, 18, 19, 24, 25, 88, 89, 90, 29, 94, 98, 101, 102, 103, 104, 105, 106, 45, 46]]) # dataset 6 su measure
        
        directory_name = str(number) + 'TablesPlots/subsamples_svm' # generate the directory for storing results
        subsample_size = 100
        subsamples = create_subsamples(directory_name, X, y, subsample_size, 100)
        print('subsamples', len(subsamples))
        number_of_test = 1    
        html = open(str(number) + 'TablesPlots/features_svm.html', 'w')
        headers = ["номер теста", "тренировочные значимые признаки", "тестовые значимые признаки", "ядро",
        "хорошие признаки выбранные one-class SVM", "полнота, точность выбора признаков one-class SVM",]
        __write_pre(html, headers)
        for sub_x, sub_y in subsamples:
            print('new subset')
            number_of_test = methodology_test(html, sub_x, sub_y, good_features[number - 1], number_of_test, kernel_parameters, scores)
        __write_post(html)
        
        fin_rec = []
        fin_prec = []
        fin_number = []
        for i in range(1, 15):
            if i in scores:
                fin_rec.append(mean(list(zip(*scores[i]))[0]))
                fin_prec.append(mean(list(zip(*scores[i]))[1]))
                fin_number.append(i)
                print('score for svm selecting ' + str(i) + ' accuracy = ' + str(mean(list(zip(*scores[i]))[1])) + ' fullness = ' + str(mean(list(zip(*scores[i]))[0])))

        fig, ax = plt.subplots(figsize=(10, 8))
        width = 0.5
        numbers = np.arange(len(fin_number))
        rects1 = ax.bar(numbers - width/2, fin_rec, width, label='Полнота')
        rects2 = ax.bar(numbers + width/2, fin_prec, width, label='Точность')
        # ax.set_ylabel('Частота отбора')
        ax.set_xlabel('Число отобранных признаков')
        ax.set_xticks(numbers)
        ax.set_xticklabels(fin_number)
        # ax.set_yticks()
        # ax.set_yticklabels([0, 25, 50])
        ax.legend()

        fig.tight_layout()

        plt.savefig(str(number) + 'TablesPlots/svm_freq.png')
        plt.close()

    # poly 1 0.0 0.67 auto  
    # poly 1 0.0 0.83 auto
    # poly 1 0.0 0.94 scale


