from functools import partial
from math import exp
from math import log

import numpy as np

def pearson_corr(X, y):
    x_dev = X - np.mean(X, axis=0)
    y_dev = y - np.mean(y, axis=0)
    sum_dev = y_dev.T.dot(x_dev)
    sq_dev_x = x_dev * x_dev
    sq_dev_y = y_dev * y_dev
    denominators = np.sqrt(np.sum(sq_dev_y, axis=0) * np.sum(sq_dev_x, axis=0))
    results = np.array([(sum_dev[i] / denominators[i]) if denominators[i] > 0.0 else 0 for i in range(len(denominators))])
    return results

GLOB_MEASURE = {"PearsonCorr": pearson_corr}


def select_best_by_value(value):
    return partial(__select_by_value, value=value, more=True)


def select_worst_by_value(value):
    return partial(__select_by_value, value=value, more=False)


def __select_by_value(scores, value, more=True):
    features = []
    for key, sc_value in scores.items():
        if more:
            if sc_value >= value:
                features.append(key)
        else:
            if sc_value <= value:
                features.append(key)
    return features


def select_k_best(k):
    return partial(__select_k, k=k, reverse=True)


def select_k_worst(k):
    return partial(__select_k, k=k)


def __select_k(scores, k, reverse=False):
    if type(k) != int:
        raise TypeError("Number of features should be integer")
    return [keys[0] for keys in sorted(scores.items(), key=lambda kv: kv[1], reverse=reverse)[:k]]


GLOB_CR = {"Best by value": select_best_by_value,
           "Worst by value": select_worst_by_value,
           "K best": select_k_best,
           "K worst": select_k_worst}