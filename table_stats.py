import numpy as np
import os
import csv
from collections import defaultdict
from math import log
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from utils import read_subsamples, create_subsamples
from utils import feature_mask
from ITMO_FS.ensembles.measure_based.MelifStable import MelifStable
from ITMO_FS.ensembles.measure_based.MelifLossFStable import MelifLossFStable
from ITMO_FS.ensembles.measure_based.Melif2PhaseStable import Melif2PhaseStable
from ITMO_FS.ensembles.measure_based.Melif import Melif
from ITMO_FS.filters.univariate import GLOB_MEASURE, pearson_corr, spearman_corr, information_gain, su_measure, chi2_measure
from ITMO_FS.filters.univariate import UnivariateFilter
from ITMO_FS.filters.univariate import select_k_best, select_best_by_value, select_k_worst
from ITMO_FS.wrappers.deterministic import SequentialForwardSelection, BackwardSelection
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from functools import partial
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import recall_score, precision_score, f1_score, jaccard_score
from sklearn.feature_selection import mutual_info_classif
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import make_scorer

starting_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
starting_dict = {i:starting_list for i in range(0, len(GLOB_MEASURE))}
basic_list = [0.1, 0.5, 1.0]
basic_dict = {i:basic_list for i in range(0, len(GLOB_MEASURE))}


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

def select_by_percentage(scores, percent):
    features = []
    max_val = max(scores.values())
    threshold = max_val * percent / 100
    for key, sc_value in scores.items():
        if sc_value >= threshold:
            features.append(key)
    return features

def methodology_test(X, y, good_features, number_of_test, scores):
    feature_split = StratifiedKFold(5, shuffle=True)
    feature_marks = feature_mask(X.shape[1], good_features)

    estimator = SVC()
    filters = list(map(lambda measure: UnivariateFilter(measure, partial(select_by_percentage, percent=80)), GLOB_MEASURE.values()))
    param_grid = starting_dict
    delta = 0.1      
    for feature_train, feature_test in feature_split.split(X.T, feature_marks):
        train_mapping = {i:f for i, f in enumerate(feature_train)}
        test_mapping = {i:f for i, f in enumerate(feature_test)}
        
        
        print('new test number:', number_of_test)
        X_ftrain = X[:, feature_train]
        X_ftest = X[:, feature_test]
        good_features_test = [value for value in test_mapping.values() if value in good_features]
        good_features_train = [value for value in train_mapping.values() if value in good_features]

        score_train_prec = partial(loss_prec, good_features=good_features, mapping=train_mapping)
        score_test_prec = partial(loss_prec, good_features=good_features, mapping=test_mapping)
        
        melif_prec = MelifLossFStable(filters, score_train_prec)
        melif_prec.fit(X_ftrain, y, 80, delta=delta, points=ParameterGrid(param_grid))
        melif_prec.run()
        feat_prec = melif_prec.transform(X_ftest, y, 80)
        sel_prec = [test_mapping[f] for f in feat_prec]
        good_prec = [test_mapping[f] for f in feat_prec if test_mapping[f] in good_features]
        print('precision passed')
        # train and test melif on f1 score            
        score_train_f1 = partial(loss_f1, good_features=good_features, mapping=train_mapping)
        score_test_f1 = partial(loss_f1, good_features=good_features, mapping=test_mapping)
        
        melif_f1 = MelifLossFStable(filters, score_train_f1)
        melif_f1.fit(X_ftrain, y, 80, delta=delta, points=ParameterGrid(param_grid))
        melif_f1.run()
        feat_f1 = melif_f1.transform(X_ftest, y, 80)
        sel_f1 = [test_mapping[f] for f in feat_f1]
        good_f1 = [test_mapping[f] for f in feat_f1 if test_mapping[f] in good_features]
        print('f1 passed')
        # train and test melif on 2phase
        score_train_2phase = partial(loss_2phase, good_features=good_features, mapping=train_mapping)
        score_test_2phase = partial(loss_2phase, good_features=good_features, mapping=test_mapping)
        
        melif_2phase = Melif2PhaseStable(filters, score_train_2phase)
        melif_2phase.fit(X_ftrain, y, 80, delta=delta, points=ParameterGrid(param_grid))
        melif_2phase.run()
        feat_2phase = melif_2phase.transform(X_ftest, y, 80)
        sel_2phase = [test_mapping[f] for f in feat_2phase]
        good_2phase = [test_mapping[f] for f in feat_2phase if test_mapping[f] in good_features]
        print('2phase passed')
        
        rec_prec, prec_prec = score_test_2phase(feat_prec)
        scores.append([rec_prec, prec_prec, 'точность'])
        rec_f1, prec_f1 = score_test_2phase(feat_f1)
        scores.append([rec_f1, prec_f1, 'F1-мера на признаках'])
        rec_2phase, prec_2phase = score_test_2phase(feat_2phase)
        scores.append([rec_2phase, prec_2phase, 'двух-фазная мера'])
        
        sample_split = StratifiedKFold(5, shuffle=True)
        for sample_train, sample_test in sample_split.split(X, y):
            
            melif_usual = MelifStable(filters, f1_score)
            melif_usual.fit(X_ftrain[sample_train], y[sample_train], estimator, X_ftrain[sample_test], y[sample_test], delta=delta, points=ParameterGrid(basic_dict))
            melif_usual.run()
            feat_usual = melif_usual.transform(X_ftest, y)
            rec_usual, prec_usual = score_test_2phase(feat_usual)
            scores.append([rec_usual, prec_usual, 'F1-мера на объетках'])
            print('basic melif passed')
            melif_ontest = MelifStable(filters, f1_score)
            melif_ontest.fit(X_ftest[sample_train], y[sample_train], estimator, X_ftest[sample_test], y[sample_test], delta=delta, points=ParameterGrid(basic_dict))
            feat_ontest = melif_ontest.run()
            rec_ontest, prec_ontest = score_test_2phase(feat_usual)
            scores.append([rec_ontest, prec_ontest, 'F1-мера только на тестовых признаках'])
            print('ontest melif passed')
            break

        for filter_ in filters:
            filter_.fit(X_ftest, y)
            sel_feat = filter_.selected_features
            rec, prec = score_test_2phase(sel_feat)
            scores.append([rec, prec, filter_.measure.__name__])
        print('filters passed')

        wrapper_forw = SequentialForwardSelection(estimator, 6, make_scorer(f1_score))
        wrapper_forw.fit(X_ftest, y)
        sel_feat = wrapper_forw.selected_features
        rec, prec = score_test_2phase(sel_feat)
        scores.append([rec, prec, 'sequential forward selection'])
        print('seq wrapper passed')

        wrapper_back = BackwardSelection(estimator, 6, make_scorer(f1_score))
        wrapper_back.fit(X_ftest, y)
        sel_feat = wrapper_back.selected_features
        rec, prec = score_test_2phase(sel_feat)
        scores.append([rec, prec, 'backward selection'])

        print('back wrapper passed')
        # train casual melif
        # score = f1_score
        # melif = Melif(filters, score)
        # melif.fit(X_ftrain[sample_train], y[sample_train], estimator, select_k_best(24), X_ftrain[sample_test], y[sample_test], delta=delta, points=ParameterGrid(param_grid))
        # melif.run()
        # feat_m = melif.transform(X_ftest[sample_train], y[sample_train], select_k_best(6))
        # sel_m = [test_mapping[f] for f in feat_m]
        # good_m = [test_mapping[f] for f in feat_m if test_mapping[f] in good_features]
        
        # goods = [sel_rec, sel_prec, sel_f1, sel_2phase, sel_m]
        # best_percentages = [melif_rec.best_percentage, melif_prec.best_percentage, melif_f1.best_percentage, melif_2phase.best_percentage]
        # best_points = [melif_rec.best_point, melif_prec.best_point, melif_f1.best_point, melif_2phase.best_point, melif.best_point]
        # scores_html = [score_test_2phase(feat_rec), score_test_2phase(feat_prec), score_test_2phase(feat_f1), score_test_2phase(feat_2phase), score_test_2phase(feat_m)]

        # goods = [sel_prec, sel_f1, sel_2phase]
        # best_percentages = [melif_prec.best_percentage, melif_f1.best_percentage, melif_2phase.best_percentage]
        # best_points = [melif_prec.best_point, melif_f1.best_point, melif_2phase.best_point]
        # scores_html = [score_test_2phase(feat_prec), score_test_2phase(feat_f1), score_test_2phase(feat_2phase)]

        # __write_row(html, number_of_test, good_features_train, good_features_test, goods, best_points, scores_html, best_percentages)
        # html.flush()

        # rec_rec, prec_rec = score_test_2phase(feat_rec)
        # scores.append([rec_rec, prec_rec, melif_rec.best_percentage, 'recall'])
        # rec_m, prec_m = score_test_2phase(feat_m)
        # scores.append([rec_m, prec_m, 'f1_object'])
        number_of_test += 1
    return number_of_test

prec_map = defaultdict(list)
rec_map = defaultdict(list)

for number in range(1, 7):
    with open(str(number) + 'TablesPlots/shuffled.csv', 'r') as fd: # open each file 
        sns.set(style="darkgrid")
        
        subsample_size = 70
        X, y = read_subsamples(fd)

        good_features = np.array([[64, 193, 194, 453, 455, 458, 203, 463, 336, 338, 24, 281, 153, 344, 472, 347, 475, 415, 35, 169, 105, 493, 378, 433, 50, 241, 442, 443, 318, 319], # dataset 1 su measure
        [128, 576, 386, 643, 899, 195, 712, 521, 15, 849, 274, 854, 664, 793, 345, 414, 287, 868, 486, 745, 621, 622, 239, 114, 819, 374, 248, 570, 251, 764], # dataset 2 su measure
        [576, 577, 324, 521, 266, 267, 268, 269, 522, 459, 460, 402, 211, 598, 351, 352, 549, 296, 431, 239, 240, 241, 379, 181, 374, 377, 378, 571, 572, 318], # dataset 3 su measure
        [64, 195, 197, 70, 74, 76, 83, 212, 85, 216, 26, 28, 158, 162, 163, 228, 229, 102, 232, 106, 237, 50, 178, 52, 252, 183, 185, 124, 254, 191], # dataset 4 su measure
        [256, 321, 66, 517, 326, 138, 398, 784, 403, 916, 83, 534, 851, 380, 40, 169, 681, 937, 744, 362, 46, 876, 560, 561, 945, 117, 376, 315, 60, 190], # dataset 5 su measure
        [1, 4, 5, 6, 8, 9, 73, 13, 14, 16, 17, 18, 19, 24, 25, 88, 89, 90, 29, 94, 98, 101, 102, 103, 104, 105, 106, 45, 46]]) # dataset 6 su measure
        
        directory_name = str(number) + 'TablesPlots/subsamples_all_table' # generate the directory for storing results
        subsamples = create_subsamples(directory_name, X, y, subsample_size, 10)
        print('subsamples', len(subsamples))
        
        if os.path.exists(str(number) + 'TablesPlots/plotspercentage') == False:
            os.mkdir(str(number) + 'TablesPlots/plotspercentage')
        # for train_percentage in [80, 85, 90, 95, 96, 97, 98, 99]:
        number_of_test = 1
        scores = []
        
        # html = open(str(number) + 'TablesPlots/percentageplots/' + str(train_percentage) + 'features_percentages_stable.html', 'w')
        # headers = ["номер теста", "признаки в тренировочной выборке", "признаки в тестовой выборке", 
        # # "good features selected by MeLiF with recall score loss", "best filter weight vector by MeLiF with recall score loss", "best percentage by Melif with recall score loss", "recall, precision score on MeLiF with recall score loss", 
        # "значимые признаки выбранные MeLiF с функцией потерь точностью", "коэффициенты выбранные MeLiF с функцией потерь точностью", "лучший процент выбранный MeLiF с функцией потерь точностью", "точность и полнота на MeLiF с функцией потерь точностью", 
        # "значимые признаки выбранные MeLiF с функцией потерь ф1 мерой на признаках", "коэффициенты выбранные MeLiF с функцией потерь ф1 мерой на признаках", "лучший процент выбранный MeLiF с функцией потерь ф1 мерой на признаках", "точность и полнота на MeLiF с функцией потерь ф1 мерой на признаках", 
        # "значимые признаки выбранные MeLiF с функцией потерь двух-фазной", "коэффициенты выбранные MeLiF с функцией потерь двух-фазной", "лучший процент выбранный MeLiF с функцией потерь двух-фазной", "точность и полнота на MeLiF с функцией потерь двух-фазной",]
        # "good features selected by MeLiF with f1 score object loss", "best filter weight vector by MeLiF with f1 score object loss", "recall, precision score on MeLiF with f1 score object loss",]
        
        # __write_pre(html, headers)
        # for test_percentage in range(80, 100):
        for sub_x, sub_y in subsamples:
            print('new subset')
            number_of_test = methodology_test(sub_x, sub_y, good_features[number - 1], number_of_test, scores)
        # __write_post(html)
        scores = np.array(scores)
        names = np.unique(scores[:, 2])
        for alg_name in names:
            alg_score = scores[np.where(scores[:, 2] == alg_name)[0]]
            
            alg_rec_mean = np.mean(alg_score[:, 0].astype(float))
            alg_rec_dev = np.std(alg_score[:, 0].astype(float))
            rec_map[alg_name].append((alg_rec_mean, alg_rec_dev))
            
            alg_prec_mean = np.mean(alg_score[:, 1].astype(float))
            alg_prec_dev = np.std(alg_score[:, 1].astype(float))
            prec_map[alg_name].append((alg_prec_mean, alg_prec_dev))
            
        # df = pd.DataFrame(data=scores, index=range(len(scores)), columns=["полнота", "точность", "процент", "функция потерь"])
        # sns.lineplot(x="процент", y="полнота",
        #          hue="функция потерь",
        #          data=df)

# prec_map = {'filter1':[(0.3458854, 0.512311), (0.19845, 0.28785), (0.4123123123, 0.55333)], 'filter2':[(0.31254, 0.12311), (0.21345, 0.55785), (0.323123123, 0.12333)]}

def write_stats(writer_name, score_map, label_name):
    writer_stats = open(writer_name + '_stat.txt', 'w')
    writer_stats.write('\\begin{table}[!h]\n')
    writer_stats.write('\\caption{Таблица сравнения алгоритмов по точности}\\label{' + label_name + ':apx}\n')
    writer_stats.write('\\centering\n')
    writer_stats.write('\\begin{tabularx}{\\textwidth}{|*{12}{>{\\centering\\arraybackslash}X|}}\\hline\n')
    writer_stats.write('& \\multicolumn{6}{|c|}{Номер набора данных}\\\\\\hline\n')

    for name, score_list in score_map.items():
        writer_stats.write(name)
        for score in score_list:
            mean_ = score[0]
            writer_stats.write(' & ' + str(round(mean_, 3)))
        writer_stats.write(' \\\\\\hline\n')
    writer_stats.write('\\end{tabularx}\n')
    writer_stats.write('\\end{table}\n')

write_stats('prec', prec_map, 'prec_comparison')
write_stats('rec', rec_map, 'rec_comparison')








