import numpy as np
import os
import random
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
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

starting_list = [0.1, 0.5, 1.0]
starting_dict = {i:starting_list for i in range(0, len(GLOB_MEASURE))}

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

def methodology_test(X, y, good_features, number_of_test, scores, k):
    feature_split = StratifiedKFold(5, shuffle=True)
    feature_marks = feature_mask(X.shape[1], good_features)

    estimator = SVC()
    filters = list(map(lambda measure: UnivariateFilter(measure, select_k_best(30)), GLOB_MEASURE.values()))
    param_grid = starting_dict
    delta = 0.1      

    for feature_train, feature_test in feature_split.split(X.T, feature_marks):
        
        feature_train_good = [f for f in feature_train if f in good_features]
        feature_test_good = [f for f in feature_test if f in good_features]
        
        # good_train_samp = random.sample(feature_train_good, k)
        good_test_samp = random.sample(feature_test_good, k)

        # print('good features train', feature_train_good)
        # print('good features train sampled', good_train_samp)

        # print('good features test', feature_test_good)
        # print('good features test sampled', good_test_samp)
        # print(sorted(feature_train))
        # print(sorted(feature_train_good))
        # feature_train_del = np.setdiff1d(feature_train, feature_train_good)
        # print('confirm deletion', set(feature_train_del).intersection(set(feature_train_good)))
        # feature_train_fin = np.append(feature_train_del, good_train_samp)
        # print('confirm append', set(feature_train_fin).intersection(set(good_train_samp)))

        feature_test_del = np.setdiff1d(feature_test, feature_test_good)
        feature_test_fin = np.append(feature_test_del, good_test_samp)

        train_mapping = {i:f for i, f in enumerate(feature_train)}
        test_mapping = {i:f for i, f in enumerate(feature_test_fin)}
        
        sample_split = StratifiedKFold(5)
        
        for sample_train, sample_test in sample_split.split(X, y):
            print('new test number:', number_of_test)
            
            X_ftrain = X[:, feature_train]
            X_ftest = X[:, feature_test_fin]
            # good_features_test = [value for value in test_mapping.values() if value in good_features]
            # good_features_train = [value for value in train_mapping.values() if value in good_features]

            # train and test melif on recall
            # score_train_rec = partial(loss_rec, good_features=good_features, mapping=train_mapping)
            # score_test_rec = partial(loss_rec, good_features=good_features, mapping=test_mapping)

            # melif_rec = MelifLossF(filters, score_train_rec)
            # melif_rec.fit(X_ftrain[sample_train], y[sample_train], select_k_best(24), delta=delta, points=ParameterGrid(param_grid))
            # melif_rec.run()
            # feat_rec = melif_rec.transform(X_ftest[sample_train], y[sample_train], select_k_best(k))
            # sel_rec = [test_mapping[f] for f in feat_rec]
            # good_rec = [test_mapping[f] for f in feat_rec if test_mapping[f] in good_features]

            # train and test melif on precision
            score_train_prec = partial(loss_prec, good_features=good_features, mapping=train_mapping)
            score_test_prec = partial(loss_prec, good_features=good_features, mapping=test_mapping)
            
            melif_prec = MelifLossF(filters, score_train_prec)
            melif_prec.fit(X_ftrain[sample_train], y[sample_train], select_k_best(24), delta=delta, points=ParameterGrid(param_grid))
            melif_prec.run()
            feat_prec = melif_prec.transform(X_ftest[sample_train], y[sample_train], select_k_best(k))
            sel_prec = [test_mapping[f] for f in feat_prec]
            good_prec = [test_mapping[f] for f in feat_prec if test_mapping[f] in good_features]

            # train and test melif on f1 score            
            # score_train_f1 = partial(loss_f1, good_features=good_features, mapping=train_mapping)
            # score_test_f1 = partial(loss_f1, good_features=good_features, mapping=test_mapping)
            
            # melif_f1 = MelifLossF(filters, score_train_f1)
            # melif_f1.fit(X_ftrain[sample_train], y[sample_train], select_k_best(24), delta=delta, points=ParameterGrid(param_grid))
            # melif_f1.run()
            # feat_f1 = melif_f1.transform(X_ftest[sample_train], y[sample_train], select_k_best(k))
            # sel_f1 = [test_mapping[f] for f in feat_f1]
            # good_f1 = [test_mapping[f] for f in feat_f1 if test_mapping[f] in good_features]

            # # train and test melif on 2phase
            # score_train_2phase = partial(loss_2phase, good_features=good_features, mapping=train_mapping)
            score_test_2phase = partial(loss_2phase, good_features=good_features, mapping=test_mapping)
            
            # melif_2phase = Melif2Phase(filters, score_train_2phase)
            # melif_2phase.fit(X_ftrain[sample_train], y[sample_train], select_k_best(24), delta=delta, points=ParameterGrid(param_grid))
            # melif_2phase.run()
            # feat_2phase = melif_2phase.transform(X_ftest[sample_train], y[sample_train], select_k_best(k))
            # sel_2phase = [test_mapping[f] for f in feat_2phase]
            # good_2phase = [test_mapping[f] for f in feat_2phase if test_mapping[f] in good_features]

            # train casual melif
            score = f1_score
            melif = Melif(filters, score)
            melif.fit(X_ftrain[sample_train], y[sample_train], estimator, select_k_best(24), X_ftrain[sample_test], y[sample_test], delta=delta, points=ParameterGrid(param_grid))
            melif.run()
            feat_m = melif.transform(X_ftest[sample_train], y[sample_train], select_k_best(k))
            sel_m = [test_mapping[f] for f in feat_m]
            good_m = [test_mapping[f] for f in feat_m if test_mapping[f] in good_features]

            # train melif straight on test features
            score = f1_score
            melif_test = Melif(filters, score)
            melif_test.fit(X_ftest[sample_train], y[sample_train], estimator, select_k_best(k), X_ftest[sample_test], y[sample_test], delta=delta, points=ParameterGrid(param_grid))
            feat_test = melif_test.run()
            sel_test = [test_mapping[f] for f in feat_test]
            good_test = [test_mapping[f] for f in feat_test if test_mapping[f] in good_features]            
            
            # goods = [sel_rec, sel_prec, sel_f1, sel_2phase, sel_m]
            # best_points = [melif_rec.best_point, melif_prec.best_point, melif_f1.best_point, melif_2phase.best_point, melif.best_point]
            # scores = [score_test_2phase(feat_rec), score_test_2phase(feat_prec), score_test_2phase(feat_f1), score_test_2phase(feat_2phase), score_test_2phase(feat_m)]

            # rec_rec, prec_rec = score_test_2phase(feat_rec)
            # scores.append([rec_rec, prec_rec, k, 'recall'])
            rec_prec, prec_prec = score_test_2phase(feat_prec)
            scores.append([rec_prec, prec_prec, k, 'точность'])
            # rec_f1, prec_f1 = score_test_2phase(feat_f1)
            # scores.append([rec_f1, prec_f1, k, 'f1_feature'])
            # rec_2phase, prec_2phase = score_test_2phase(feat_2phase)
            # scores.append([rec_2phase, prec_2phase, k, '2phase'])
            rec_m, prec_m = score_test_2phase(feat_m)
            scores.append([rec_m, prec_m, k, 'ф1 мера на объектах'])
            rec_test, prec_test = score_test_2phase(feat_test)
            scores.append([rec_test, prec_test, k, 'на тестовых признаках'])

            number_of_test += 1
    return number_of_test

for number in range(6, 7):
    good_features = np.array([[64, 193, 194, 453, 455, 458, 203, 463, 336, 338, 24, 281, 153, 344, 472, 347, 475, 415, 35, 169, 105, 493, 378, 433, 50, 241, 442, 443, 318, 319], # dataset 1 su measure
    [128, 576, 386, 643, 899, 195, 712, 521, 15, 849, 274, 854, 664, 793, 345, 414, 287, 868, 486, 745, 621, 622, 239, 114, 819, 374, 248, 570, 251, 764], # dataset 2 su measure
    [576, 577, 324, 521, 266, 267, 268, 269, 522, 459, 460, 402, 211, 598, 351, 352, 549, 296, 431, 239, 240, 241, 379, 181, 374, 377, 378, 571, 572, 318], # dataset 3 su measure
    [64, 195, 197, 70, 74, 76, 83, 212, 85, 216, 26, 28, 158, 162, 163, 228, 229, 102, 232, 106, 237, 50, 178, 52, 252, 183, 185, 124, 254, 191], # dataset 4 su measure
    [256, 321, 66, 517, 326, 138, 398, 784, 403, 916, 83, 534, 851, 380, 40, 169, 681, 937, 744, 362, 46, 876, 560, 561, 945, 117, 376, 315, 60, 190], # dataset 5 su measure
    [1, 4, 5, 6, 7, 8, 9, 73, 13, 14, 16, 17, 18, 19, 24, 25, 88, 89, 90, 29, 94, 98, 101, 102, 103, 104, 105, 106, 45, 46]]) # dataset 6 su measure
    subsample_size = 70
    select_k_number = list(range(1, 7)) 
    with open(str(number) + 'TablesPlots/shuffled.csv', 'r') as fd: # open each file 
        sns.set(style="darkgrid")
        X, y = read_subsamples(fd)
        directory_name = str(number) + 'TablesPlots/subsamples_melif' # generate the directory for storing results
        subsamples = create_subsamples(directory_name, X, y, subsample_size, 5)
        print('subsamples', len(subsamples))
        number_of_test = 1
        scores = []
        for sub_x, sub_y in subsamples:
            print('new subset')
            for k in select_k_number:
                number_of_test = methodology_test(sub_x, sub_y, good_features[number - 1], number_of_test, scores, k)
        
        dump_points = open(str(number) + 'TablesPlots/dump_points_variate_test.txt', 'w')
        dump_points.write(str(len(scores)) + ' ' + str(len(scores[0])))
        for i in range(len(scores)):
            for j in range(len(scores[i])):
                dump_points.write(str(scores[i][j]) + ' ')
            dump_points.write('\n')

        df1 = pd.DataFrame(data=scores, index=range(len(scores)), columns=["полнота", "точность", "число тестирующих признаков", "функция потерь"])
        g = sns.lineplot(x="число тестирующих признаков", y="точность",
                 hue="функция потерь",
                 data=df1)
        ax = g.axes
        ax.set_xticklabels(range(0, 7)) # set new labels
        plt.savefig(str(number) + 'TablesPlots/scores_prec_variate_test.png')
        plt.close()
        
        df2 = pd.DataFrame(data=scores, index=range(len(scores)), columns=["полнота", "точность", "число тестирующих признаков", "функция потерь"])
        g = sns.lineplot(x="число тестирующих признаков", y="полнота",
                 hue="функция потерь",
                 data=df2)
        ax = g.axes
        ax.set_xticklabels(range(0, 7)) # set new labels
        plt.savefig(str(number) + 'TablesPlots/scores_rec_variate_test.png')
        plt.close()



