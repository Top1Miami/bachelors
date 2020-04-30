from sklearn.svm import OneClassSVM
import numpy as np

class SemiSupervisedFeatureSelection(object):

    def __init__(self):
        pass

    def run(self, x, known_features):
        train_set = x[:, known_features].T
        clf = OneClassSVM(nu = 1/x.shape[1], kernel='poly', degree=2, gamma='scale')
        clf.fit(train_set)
        test_set = np.delete(x, known_features, axis=1)
        self.selected_features = clf.predict(test_set.T)

class SemiSupervisedFeatureSelectionGreedy(object):

    def __init__(self, kernel_paramteres):
        params = kernel_paramteres.split(' ')
        if params[0] == 'linear':
            self.clf = OneClassSVM(kernel='linear', nu = float(params[1]))
        elif params[0] == 'rbf':
            self.clf = OneClassSVM(kernel='rbf', nu = float(params[1]), gamma = params[2])
        elif params[0] == 'sigmoid':
            self.clf = OneClassSVM(kernel='sigmoid', coef0 = float(params[1]), nu = float(params[2]), gamma = params[3])
        else:
            self.clf = OneClassSVM(kernel='poly', degree = int(params[1]), coef0 = float(params[2]), nu = float(params[3]), gamma = params[4])

    def run(self, x, known_features):
        train_set = x[:, known_features].T
        self.clf.fit(train_set)
        test_set = np.delete(x, known_features, axis=1)
        self.selected_features = self.clf.predict(test_set.T)