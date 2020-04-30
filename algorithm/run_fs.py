import numpy as np
from .html_printer import html_print
import heapq
import measures
from .SemiSupervisedFs import SemiSupervisedFeatureSelection
from .SemiSupervisedFs import SemiSupervisedFeatureSelectionGreedy
from .metrics import compare_results

# def __load_cfg():
#     with open('experiments.cfg', 'r') as fd:
#         file_list = fd.readlines()
#         file_list = list(map(lambda x: x[:-1], file_list))
#         number_of_datasets = int(file_list[0])
#         file_list.pop(0)
#         kernel_parameters = file_list[file_number - 1].split(' ')
#     return kernel_parameters

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

def compare_with_greedy(part_x, part_y, known_features, good_features, kernel_parameters):
    pearson_selected = measures.pearson_corr(part_x, part_y)
    baseline_features = list(zip(*sorted(zip(np.arange(1, pearson_selected.size + 1), pearson_selected), key=lambda kv: kv[1], reverse=True)[:30]))[0]
    # kernel_parameters = __load_cfg()
    best_result = 0.0, 0.0
    best_config = ""
    for i, single_kernel in enumerate(kernel_parameters):
        fs_alg = SemiSupervisedFeatureSelectionGreedy(single_kernel)
        fs_alg.run(part_x, (known_features - 1))
        semi_features = np.where(fs_alg.selected_features == 1)[0] + 1
        cur_results = compare_results(semi_features, baseline_features, known_features, good_features)
        if cur_results[0] > 0.2 and cur_results[1] > 0.2: 
            print("know:", known_features)
            print("good:", good_features)
            print("baseline:", baseline_features)
            print("semi:", semi_features)
            print(cur_results)    
            # if best_result[0] <= cur_results[0] and best_result[1] < cur_results[1]:
            #     best_result = cur_results[0] , cur_results[1]
            #     best_config = single_kernel
    return best_result, best_config

def run_greedy(part_x, part_y, known_features, good_features):
    kernel_parameters = []
    for degree in range(1, 7):
        for coef in range(1, 10):
            coef0 = float(coef) / 10     
            for nu in range(1, 10):
                nu0 = float(nu) / 100
                kernel_parameters.append('poly ' + str(degree) + ' ' + str(coef0) + ' ' + str(nu0) + ' scale')
                kernel_parameters.append('poly ' + str(degree) + ' ' + str(coef0) + ' ' + str(nu0) + ' auto')
        for coef in range(2, 10):
            for nu in range(1, 10):
                nu0 = float(nu) / 100
                kernel_parameters.append('poly ' + str(degree) + ' ' + str(coef) + ' ' + str(nu0) + ' scale')
                kernel_parameters.append('poly ' + str(degree) + ' ' + str(coef) + ' ' + str(nu0) + ' auto')


    for coef in range(1, 100):
        coef0 = float(coef) / 100    
        for nu in range(1, 100):
            nu0 = float(nu) / 100
            kernel_parameters.append('sigmoid ' + str(coef0) + ' ' + str(nu0) + ' scale')
            kernel_parameters.append('sigmoid ' + str(coef0) + ' ' + str(nu0) + ' auto')
    for coef in range(100, 1000):
        for nu in range(1, 100):
            coef0 = float(coef) / 100
            nu0 = float(nu) / 100
            kernel_parameters.append('sigmoid ' + str(coef0) + ' ' + str(nu0) + ' scale')
            kernel_parameters.append('sigmoid ' + str(coef0) + ' ' + str(nu0) + ' auto')

    for nu in range(1, 10):
        nu0 = float(nu) / 100
        kernel_parameters.append('rbf ' + str(nu0) + ' scale')
        kernel_parameters.append('rbf ' + str(nu0) + ' auto')
        kernel_parameters.append('linear ' + str(nu0))
        kernel_parameters.append('linear ' + str(nu0))

    result = compare_with_greedy(part_x, part_y, known_features, good_features, kernel_parameters)
    print(result)


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
        