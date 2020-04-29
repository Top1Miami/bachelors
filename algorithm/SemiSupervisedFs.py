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