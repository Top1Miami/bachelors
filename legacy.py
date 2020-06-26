# good_features = np.array([414, 248, 386, 251, 345, 521, 576, 643, 712, 819, 15, 114, 128, 239, 287, 374, 486, 621, 622, 664, 745, 793, 849, 854, 868, 899, 274, 570, 764, 923]) # info gain
    # good_features = np.array([932, 387, 675, 129, 642, 322, 388, 836, 509, 391, 461, 208, 726, 668, 92, 417, 803, 676, 933, 167, 938, 108, 173, 110, 879, 177, 753, 757, 501, 189, 120, 573, 766]) # pearson
    # kfold = KFold(5)
    # tr_ts_split = StratifiedShuffleSplit(1, test_size=0.2, random_state=0)
    # estimator = SVC()
    # for m in [su_measure, chi2_measure]:
    #     print(m.__name__)
    #     for train_set, test_set in tr_ts_split.split(X, y):
    #         X_tr = X[train_set]
    #         y_tr = y[train_set]
    #         for train, test in kfold.split(X_tr, y_tr):
    #             unv_filter = UnivariateFilter(m, select_k_best(30))
    #             unv_filter.fit(X_tr[train], y_tr[train])
    #             estimator.fit(X_tr[train], y_tr[train])
    #             predicted = estimator.predict(X_tr[test])
    #             score = f1_score(y_tr[test], predicted)
    #             print('no fs', score)
    #             sel_feat = unv_filter.selected_features
    #             estimator.fit(X_tr[train][:, sel_feat], y_tr[train])
    #             predicted = estimator.predict(X_tr[test][:, sel_feat])
    #             score = f1_score(y_tr[test], predicted)
    #             print('fs', score)
    #         unv_filter = UnivariateFilter(m, select_k_best(30))
    #         unv_filter.fit(X[train_set], y[train_set])
    #         estimator.fit(X[train_set], y[train_set])
    #         predicted = estimator.predict(X[test_set])
    #         score = f1_score(y[test_set], predicted)
    #         print('no fs final', score)
    #         estimator.fit(X[train_set][:, sel_feat], y[train_set])
    #         predicted = estimator.predict(X[test_set][:, sel_feat])
    #         score = f1_score(y[test_set], predicted)
    #         print('fs final', score)
    #         print(sorted(sel_feat))
    #         print(sorted(good_features))
    