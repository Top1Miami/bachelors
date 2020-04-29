import numpy as np
from .html_printer import html_print
import heapq
import measures
from .SemiSupervisedFs import SemiSupervisedFeatureSelection
from .metrics import compare_results

def __load_cfg():
    with open('experiments.cfg', 'r') as fd:
        file_list = fd.readlines()
        file_list = list(map(lambda x: x[:-1], file_list))
        number_of_datasets = int(file_list[0])
        file_list.pop(0)
        kernel_parameters = file_list[file_number - 1].split(' ')
    return kernel_parameters

def compare_with(part_x, part_y, known_features, good_features):
    fs_alg = SemiSupervisedFeatureSelection()
    fs_alg.run(part_x, (known_features - 1))
    pearson_selected = measures.pearson_corr(part_x, part_y)
    baseline_features = list(zip(*sorted(zip(np.arange(1, pearson_selected.size + 1), pearson_selected), key=lambda kv: kv[1], reverse=True)[:30]))[0]
    semi_features = np.where(fs_alg.selected_features == 1)[0] + 1
    # print('good:', sorted(good_features), 'len =', len(good_features))
    # print('known:', sorted(known_features), 'len =', len(known_features))
    # print('semi:', sorted(semi_features), 'len =', len(semi_features))
    # print('baseline:', sorted(baseline_features), 'len =', len(baseline_features))
    # print()
    print(compare_results(semi_features, baseline_features, known_features, good_features))
    # return pearson_selected, fs_alg.selected_features

def compare_with_greedy(part_x, part_y, known_features, good_features):
    pearson_selected = measures.pearson_corr(part_x, part_y)
    baseline_features = list(zip(*sorted(zip(np.arange(1, pearson_selected.size + 1), pearson_selected), key=lambda kv: kv[1], reverse=True)[:30]))[0]
    kernel_parameters = __load_cfg()
    for i, single_kernel in enumerate(kernel_parameters):
        fs_alg = SemiSupervisedFeatureSelectionGreedy(single_kernel)
        fs_alg.run(part_x, (known_features - 1))
        semi_features = np.where(fs_alg.selected_features == 1)[0] + 1
        print(compare_results(semi_features, baseline_features, known_features, good_features))
    return pearson_selected, fs_alg.selected_features

def run_and_compare_fs(x, y, bad_subsamples, known_features, good_features, directory_name):
    with open(directory_name + '/comparison.html', 'w') as fd:
        heap = bad_subsamples
        sorted_stacked_list = []
        while(len(heap) > 0):
            _, index_list = heapq.heappop(heap)
            part_x = x[index_list] #TODO error occured
            part_y = y[index_list]
            result = compare_with(part_x, part_y, known_features, good_features)
        #     stacked = np.stack((np.arange(1, x.shape[1] + 1, dtype=np.integer), result[0], result[1]), axis=1)
        #     sorted_stacked = np.array(sorted(stacked, key=lambda t: t[1], reverse=True))
        #     sorted_stacked_list.append(sorted_stacked)
        # html_print(fd, sorted_stacked_list, known_features)
        