import numpy as np
from .html_printer import html_print
import copy
import random
from .UnivariateFilter import UnivariateFilter
import measures
import heapq
from sklearn.model_selection import StratifiedShuffleSplit

def __load_cfg(file_number):
    with open('experiments.cfg', 'r') as fd:
        file_list = fd.readlines()
        file_list = list(map(lambda x: x[:-1], file_list))
        number_of_datasets = int(file_list[0])
        file_list.pop(0)
        range_list = file_list[file_number - 1].split(' ')
    return list(map(int, range_list))

def get_top(count_pos, k):
    count_top_k = []
    for i in count_pos:
        zipped = list(zip(np.arange(0, len(i), dtype=np.integer), i))
        count_top_k.append(sorted(zipped, key = lambda t: t[1], reverse = True)[:k])
    return np.array(count_top_k, dtype = np.integer)

def select_best(count_pos, k, part_x, part_y, slice_index):
    univ_filter = UnivariateFilter(measures.pearson_corr, measures.select_k_best(k))
    univ_filter.fit(part_x, part_y)
    sf = univ_filter.selected_features
    for f in sf:
        count_pos[slice_index][f] += 1

def get_shuffle(number_objects_by_class, split, slice_size):
    count_objects_slice = number_objects_by_class * slice_size
    count_objects_slice = count_objects_slice.astype(np.integer)
    indexes = []
    for class_number, number_of_objects in enumerate(count_objects_slice):
        for _ in range(number_of_objects):
            rand = random.randint(0, len(split[class_number]) - 1)
            indexes.append(split[class_number][rand])
            split[class_number].pop(rand)
    return indexes

def count_confidence(top_3, top_10, top_30, number_of_shuffles):
    confidence_list = []
    check_top_3 = list(zip(*top_3))[0]
    check_top_10 = list(zip(*top_10))[0]
    for i in range(30):
        conf = 0
        if top_30[i][0] in check_top_3:
            conf += top_3[check_top_3 == top_30[i][0]][0][1] / number_of_shuffles * 0.5
        if top_30[i][0] in check_top_10:
            conf += top_10[check_top_10 == top_30[i][0]][0][1] / number_of_shuffles * 0.3
        confidence_list.append((top_30[i][0], top_30[i][1] / number_of_shuffles * 0.2 + conf))
    confidence_list = sorted(confidence_list, key = lambda x: x[1], reverse = True)
    return confidence_list

def split_by(y):
    tempY = list(y.copy())
    classNumber = max(y)
    split = []
    for i in range(0, classNumber + 1):
        split.append([z for z, value in enumerate(tempY) if value == i])
    return split

def run_build_model(x, y, directory_name):
    file_number = int(directory_name[0])
    range_list = __load_cfg(file_number)
    
    objects_by_class = np.histogram(y, bins=max(y) + 1)[0] / len(y) # building histogram for counting objects by class distribution
    split_by_class = split_by(y) # samples split by class labels
    class_number = max(y) # getting class size
    feature_number = x.shape[1] # number of features

    # topTable = TopTable()
    html = open(directory_name + '/HtmlTable.html', 'w') # open file for storing feature ranks by slices
    count_pos_3 = np.zeros((len(range_list), feature_number)) # initilizing tables for top feature storing
    count_pos_10 = np.zeros((len(range_list), feature_number))
    count_pos_30 = np.zeros((len(range_list), feature_number))
    
    #TODO add stratified KFold
    number_of_shuffles = 50 # number of top feature calculation
    for slice_index, slice_size in enumerate(range_list):
        shuffler = StratifiedShuffleSplit(number_of_shuffles, test_size=slice_size, random_state=0)
        for _, shuffle_indexes in shuffler.split(x, y):
            part_x, part_y = x[shuffle_indexes], y[shuffle_indexes] # x, y subsample
            select_best(count_pos_3, 3, part_x, part_y, slice_index) # select 3 best features
            select_best(count_pos_10, 10, part_x, part_y, slice_index) # select 10 best features
            select_best(count_pos_30, 30, part_x, part_y, slice_index) # select 30 best features
    
    count_top_3 = get_top(count_pos_3, 3) # sort and cut 3 best by number of occurances 
    count_top_10 = get_top(count_pos_10, 10) # sort and cut 10 best by number of occurances 
    count_top_30 = get_top(count_pos_30, 30) # sort and cut 30 best by number of occurances 
    
    conf_by_slice = [count_confidence(count_top_3[i], count_top_10[i], count_top_30[i], number_of_shuffles) for i in range(len(range_list))]# count confidence for slices
    
    html_print(html, count_top_3, count_top_10, count_top_30, conf_by_slice, range_list)# print html

    best_conf = np.zeros(feature_number, dtype=np.integer)
    for i in range(len(conf_by_slice)):
        for j in range(0, 5):
            best_conf[conf_by_slice[i][j][0] - 1] += 1

    known_features = np.argsort(best_conf)[::-1][:3]
    good_features = list(set(count_top_30[count_top_30.shape[0] - 1][:, 0].ravel()).difference(set(known_features)))
    
    heap = []
    # shuffler = StratifiedShuffleSplit(1000, test_size=slice_size, random_state=0)
    # for _, shuffle_indexes in shuffler.split(x, y):
    #     part_x = x[shuffle_indexes] # x subsample
    #     part_y = y[shuffle_indexes] # y subsample
    #     univ_filter = UnivariateFilter(measures.pearson_corr, measures.select_k_best(30)) # create univariative filter with cutting rule 10 best
    #     univ_filter.fit(part_x, part_y) # fit the feature ranking model
    #     number_of_known_features = len(set(univ_filter.selected_features).intersection(known_features))
    #     if len(heap) > 0:
    #         lowest, _ = heapq.nsmallest(1, heap)[0]
    #         if len(heap) == 10 and -lowest > number_of_known_features:
    #             heapq.heappop(heap)
    #             heapq.heappush(heap, (-number_of_known_features, shuffle_indexes))
    #         elif len(heap) < 10:
    #             heapq.heappush(heap, (-number_of_known_features, shuffle_indexes))
    #     else:
    #         heapq.heappush(heap, (-number_of_known_features, shuffle_indexes))
    return heap, known_features, good_features
    