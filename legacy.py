with open('4TablesPlots/shuffled.csv', 'r') as fd: # open each file 
    #run_build_plots(x, y) # build all the plots for datasets
    X, y = read_subsamples(fd)
    kfold = KFold(10)
    tr_ts_split = StratifiedShuffleSplit(1, test_size=0.99, random_state=0)
    estimator = SVC()
    for m in [spearman_corr, pearson_corr]:
        print(m.__name__)
        for train_set, test_set in tr_ts_split.split(X, y):
            X_tr = X[train_set]
            y_tr = y[train_set]
            for train, test in kfold.split(X_tr, y_tr):
                unv_filter = UnivariateFilter(m, select_k_best(30))
                unv_filter.fit(X_tr[train], y_tr[train])
                estimator.fit(X_tr[train], y_tr[train])
                predicted = estimator.predict(X_tr[test])
                score = f1_score(y_tr[test], predicted)
                print('no fs', score)
                sel_feat = unv_filter.selected_features
                estimator.fit(X_tr[train][:, sel_feat], y_tr[train])
                predicted = estimator.predict(X_tr[test][:, sel_feat])
                score = f1_score(y_tr[test], predicted)
                print('fs', score)
            unv_filter = UnivariateFilter(m, select_k_best(30))
            unv_filter.fit(X[train_set], y[train_set])
            estimator.fit(X[train_set], y[train_set])
            predicted = estimator.predict(X[test_set])
            score = f1_score(y[test_set], predicted)
            print('no fs final', score)
            estimator.fit(X[train_set][:, sel_feat], y[train_set])
            predicted = estimator.predict(X[test_set][:, sel_feat])
            score = f1_score(y[test_set], predicted)
            print('fs final', score)