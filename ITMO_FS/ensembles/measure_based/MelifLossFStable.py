import datetime as dt
import logging

import numpy as np
from sklearn.model_selection import train_test_split

from ITMO_FS.utils.data_check import *
from sklearn.model_selection import ParameterGrid

class MelifLossFStable:
    __filters = [] # filters used for composition
    __feature_names = [] # in case features are named 
    __filter_weights = [] # weights of the filters in composition
    __points = [] # list of filter weights walked through while reaching the best solution
    __delta = None # the value for filter_weight to be increased or dicreased by
    _train_x = _train_y = None # the training and testing parts of the dataset
    
    def __init__(self, filters, score=None):  # TODO scorer name
        # check_filters(filters)
        self.__filters = filters # filters used for composition
        self.__score = score # score metric for evaluting feature selection quality
        self.best_score = 0 # best accumulated score during the work of Melif
        self.best_point = np.array([1/len(filters) for _ in range(len(filters))]) # best weight vector for filters in composition
        self.best_f = None

    def select_by_percentage(self, scores, percent):
        features = []
        max_val = max(scores.values())
        threshold = max_val * percent / 100
        for key, sc_value in scores.items():
            if sc_value >= threshold:
                features.append(key)
        return features

    def fit(self, X, y, percentage, delta=0.5, feature_names=None, points=None):
        """
        TODO comments
        :param X:
        :param y:
        :param feature_names:
        :param points:
        :return:
        """
        # logging.info('Running basic MeLiF\nFilters:{}'.format(self.__filters))
        check_shapes(X, y) # check if the shapes of the given datasets are appropriate
        self.__feature_names = generate_features(X, feature_names) # initialize the feature names
        self.__filter_weights = np.ones(len(self.__filters)) / len(self.__filters) # initialize the weights with starting value
        self.__points = points # list of points to start with if given
        self.__delta = delta # delta if given by default 0.5 is used 
        self.__percentage = percentage # percentage for cutting rule
        # logging.info(
        #     "Optimizer greedy search, optimizing measure is {}".format(
        #         self.__score))  # TODO add optimizer and quality measure
        # time = dt.datetime.now() 
        # logging.info("time:{}".format(time))

        self._train_x, self._train_y = X, y # training and testing datasets


    def run(self): # run the melif feature selecting algorithm
        """
        TODO comments
        :param cutting_rule:
        :param test_size:
        :param delta:
        :return:
        """

        nu = {i: [] for i in self.__feature_names} # dictionary for list of feature scores
        for _filter in self.__filters: # iterate through filters
            _filter.fit_transform(self._train_x, self._train_y,
                                  feature_names=self.__feature_names, store_scores=True) # fit the given filters
            for key, value in _filter.feature_scores.items(): # iterate through feature scores
                _filter.feature_scores[key] = abs(value) # take the absolute values as some filters such as pearson
                # have negative scores showing the dependency between features and labels 
            _min = min(_filter.feature_scores.values()) # take the minimum score value
            _max = max(_filter.feature_scores.values()) # take the maximum score value
            for key, value in _filter.feature_scores.items(): # iterate through filter scores
                nu[key].append((value - _min) / (_max - _min)) # append to feature score list normalized scores 
        if self.__points is None: # if no points are given
            self.__points = [self.__filter_weights] # start with initial weights
        if isinstance(self.__points, ParameterGrid):
            self.__points = map(lambda d: list(d.values()), list(self.__points))
        # for point in self.__points:
        #     self.__search(point, nu) # perform the search for best filter weights
        self.__search(self.__points, nu)
        # logging.info('Footer')
        # logging.info("Best point:{}".format(self.best_point))
        # logging.info("Best Score:{}".format(self.best_score))
        # logging.info('Top features:')
        # for key, value in sorted(self.best_f.items(), key=lambda x: x[1], reverse=True):
            # logging.info("Feature: {}, value: {}".format(key, value))

        return self.best_f # return selected features

    def __search(self, points, features): # search for feature weights
        # time = dt.datetime.now()
        for cur_point in points:
            # print(cur_point)
            # print(cur_point)
            # logging.info('Time:{}'.format(dt.datetime.now() - time))
            # logging.info('point:{}'.format(point))
            values = list(features.values()) # feature score lists
            n = dict(zip(features.keys(), self.__measure(np.array(values), cur_point))) # calculate the composed score of the feature
            # logging.info(
            #     'Score at current point : {}'.format(score))
            # print('percentage', percentage)
            keys = self.select_by_percentage(n, self.__percentage) # select the features according to the composed scores
            new_features = {i: features[i] for i in keys} # dicitionary selected feature number -> its original name
            score = self.__score(keys)
            
            if score > self.best_score: # if current score is better
                self.best_score = score # save current score as best
                self.best_point = cur_point # save current filter weight vector as best
                self.best_f = new_features  # save selected features as best

    # def __search(self, point, features): # search for feature weights
    #     # time = dt.datetime.now()
    #     points = [point] # points to evaluate
    #     while(len(points) > 0): # walk through the points
    #         cur_point = points.pop() # take last added point
    #         # logging.info('Time:{}'.format(dt.datetime.now() - time))
    #         # logging.info('point:{}'.format(point))
    #         values = list(features.values()) # feature score lists
    #         n = dict(zip(features.keys(), self.__measure(np.array(values), cur_point))) # calculate the composed score of the feature
    #         # logging.info(
    #         #     'Score at current point : {}'.format(score))
    #         for percentage in range(80, 100):
    #             # print('percentage', percentage)
    #             keys = self.select_by_percentage(n, percentage) # select the features according to the composed scores
    #             new_features = {i: features[i] for i in keys} # dicitionary selected feature number -> its original name
    #             score = self.__score(keys)
                
    #             if score > self.best_score: # if current score is better
    #                 self.best_score = score # save current score as best
    #                 self.best_point = cur_point # save current filter weight vector as best
    #                 self.best_f = new_features  # save selected features as best
    #                 self.best_percentage = percentage
    #                 for candidate in self.__get_candidates(cur_point, self.__delta): # get other candidates for best solution
    #                     if candidate not in points:
    #                         points.append(candidate)

    def __get_candidates(self, point, delta=0.1):
        tiled_points = np.tile(point, (len(point) * 2, 1))
        stacked = np.vstack((np.eye(len(point)) * delta, np.eye(len(point)) * -delta))
        for i in range(tiled_points.shape[0]):
            for j in range(tiled_points.shape[1]):
                if tiled_points[i][j] + stacked[i][j] < 0.:
                    tiled_points[i][j] = 0.
                elif tiled_points[i][j] + stacked[i][j] > 1.0:
                    tiled_points[i][j] = 1.0
                else:
                    tiled_points[i][j] += stacked[i][j]
        return tiled_points.tolist()

    def __measure(self, nu, weights):
        return np.dot(nu, weights)

    def transform(self, X, y, percentage, feature_names=None): # select features according to current filter weight vector
        features = generate_features(X, feature_names) # generate feature names if none given
        
        nu = {i: [] for i in features} # dictionary for list of feature scores
        for _filter in self.__filters: # iterate through filters
            _filter.fit(X, y, feature_names, store_scores=True) # fit the given filters
            for key, value in _filter.feature_scores.items(): # iterate through feature scores
                _filter.feature_scores[key] = abs(value) # take the absolute values as some filters such as pearson
                # have negative scores showing the dependency between features and labels 
            _min = min(_filter.feature_scores.values()) # take the minimum score value
            _max = max(_filter.feature_scores.values()) # take the maximum score value
            for key, value in _filter.feature_scores.items(): # iterate through filter scores
                nu[key].append((value - _min) / (_max - _min)) # append to feature score list normalized scores
        n = dict(zip(nu.keys(), self.__measure(np.array(list(nu.values())), self.best_point))) # calculate the composed score of the feature
        
        keys = self.select_by_percentage(n, percentage) # select the features according to the composed scores
        
        new_features = {i: nu[i] for i in keys} # dictionary selected feature number -> its original name  
        return list(new_features.keys())

