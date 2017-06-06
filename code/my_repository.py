# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV

def weighted_average(Y, weights):
    # Y.shape = (weights, len(data))
    Y = np.array(Y).T
    weights = np.array(weights / np.sum(weights))
    return np.sum(Y * weights, axis=1)

def rmsle(y_true, y_pred):
    n = len(y_true)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    parcels = [(np.log(y_pred[i] + 1) - np.log(y_true[i] + 1))**2 for i in xrange(n)]
    return np.sqrt(np.sum(parcels) / n)
    
def rmse(y_true, y_pred):
    n = len(y_true)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    parcels = [(y_pred[i] - y_true[i])**2 for i in xrange(n)]
    return np.sqrt(np.sum(parcels) / n)
    
def test_model(model_def, X, y, kf):
    i = 1
    mean_score_train = 0
    mean_score_valid = 0
    for train_idx, valid_idx in kf:
        x_train, y_train = X.loc[train_idx], y.loc[train_idx]
        x_valid, y_valid = X.loc[valid_idx], y.loc[valid_idx]
        model = model_def.fit(x_train, y_train)
        y_train_pred = model.predict(x_train)
        y_valid_pred = model.predict(x_valid)
        score_train = rmsle(y_train, y_train_pred)
        score_valid = rmsle(y_valid, y_valid_pred)
        print 'Fold no.', i
        print 'Training table score (RMSLE): {0:.5f}'.format(score_train)
        print 'Validation table score (RMSLE): {0:.5f}'.format(score_valid)
        print ' '
        mean_score_train = ((i-1) * mean_score_train + score_train) / i
        mean_score_valid = ((i-1) * mean_score_valid + score_valid) / i
        i += 1
    model = model_def.fit(X, y)
    y_full_pred = model.predict(X)
    full_data_score = rmsle(y, y_full_pred)
    print 'Mean training score (RMSLE): {0:.5f}'.format(mean_score_train)
    print 'Mean validation score (RMSLE): {0:.5f}'.format(mean_score_valid)
    print ' '
    print 'Full data score (RMSLE): {0:.5F}'.format(full_data_score)
    return model, mean_score_valid
    
def test_model_array(model_defs, X, y, kf, y_type=None, w_function=None):
    n_models = len(model_defs)
    fold = 1
    mean_score_train = np.zeros(n_models)
    mean_score_valid = np.zeros(n_models)
    mean_weighted_score_train = mean_weighted_score_valid = 0
    if y_type == 'log1p':
        mean_score_log_train = mean_score_log_valid = 0
        mean_score_log_t = np.zeros(n_models)
        mean_score_log_v = np.zeros(n_models)
    for train_idx, valid_idx in kf:
        x_train, y_train = X.loc[train_idx], y.loc[train_idx]
        x_valid, y_valid = X.loc[valid_idx], y.loc[valid_idx]
        models = []
        y_train_pred = []
        y_valid_pred = []
        score_train = []
        score_valid = []
        if y_type == 'log1p':        
            score_log_t = []
            score_log_v = []
        for m_def in model_defs:
            models.append(m_def.fit(x_train, y_train))
        for model in models:
            y_train_pred.append(model.predict(x_train))
            y_valid_pred.append(model.predict(x_valid))
            score_train.append(rmsle(y_train, y_train_pred[-1]))
            score_valid.append(rmsle(y_valid, y_valid_pred[-1]))
            if y_type == 'log1p':
                score_log_t.append(rmse(y_train, y_train_pred[-1]))
                score_log_v.append(rmse(y_valid, y_valid_pred[-1]))
        if y_type == 'log1p':
            if w_function == 'inv_sum' or w_function == None:
                weights = [1 / score for score in score_log_v]
            elif w_function == 'inv_sq_sum':
                weights = [1 / score**2 for score in score_log_v]
        else:
            if w_function == 'inv_sum' or w_function == None:
                weights = [1 / score for score in score_valid]
            elif w_function == 'inv_sq_sum':
                weights = [1 / score**2 for score in score_valid]
        weights = np.array(weights) / np.sum(weights)
        weighted_score_train = rmsle(y_train, weighted_average(y_train_pred, weights))
        weighted_score_valid = rmsle(y_valid, weighted_average(y_valid_pred, weights))
        print 'Fold no.', fold
        for n in range(n_models):
            print '-- Model ' + str(n) + ' --'
            print 'Training table score (RMSLE): {0:.5f}'.format(score_train[n])
            print 'Validation table score (RMSLE): {0:.5f}'.format(score_valid[n])
            if y_type == 'log1p':
                print 'Training table score (RMSLE) for correct (exp) data: {0:.5f}'.format(score_log_t[n])
                print 'Validation table score (RMSLE) for correct (exp) data: {0:.5f}'.format(score_log_v[n])
                mean_score_log_t[n] = ((fold-1) * mean_score_log_t[n] + score_log_t[n]) / fold
                mean_score_log_v[n] = ((fold-1) * mean_score_log_v[n] + score_log_v[n]) / fold
            mean_score_train[n] = ((fold-1) * mean_score_train[n] + score_train[n]) / fold
            mean_score_valid[n] = ((fold-1) * mean_score_valid[n] + score_valid[n]) / fold
        print '--'
        print 'Weighted training table score (RMSLE): {0:.5f}'.format(weighted_score_train)
        print 'Weighted validation table score (RMSLE): {0:.5f}'.format(weighted_score_valid)
        if y_type == 'log1p':
            score_t = rmse(y_train, weighted_average(y_train_pred, weights))
            score_v = rmse(y_valid, weighted_average(y_valid_pred, weights))
            print 'Weighted training table score (RMSLE) for correct (exp) data: {0:.5f}'.format(score_t)
            print 'Weighted validation table score (RMSLE) for correct (exp) data: {0:.5f}'.format(score_v)
            mean_score_log_train = ((fold-1) * mean_score_log_train + score_t) / fold
            mean_score_log_valid = ((fold-1) * mean_score_log_valid + score_v) / fold
        print 'Current weights:', weights
        print ' '
        mean_weighted_score_train = ((fold-1) * mean_weighted_score_train + weighted_score_train) / fold
        mean_weighted_score_valid = ((fold-1) * mean_weighted_score_valid + weighted_score_valid) / fold
        fold += 1
    models = []
    y_full_pred = []
    for m_def in model_defs:
        models.append(m_def.fit(X, y))
    for model in models:
        y_full_pred.append(model.predict(X))
    if y_type == 'log1p':
        if w_function == 'inv_sum' or w_function == None:
            weights = [1 / score for score in mean_score_log_v]
        elif w_function == 'inv_sq_sum':
            weights = [1 / score**2 for score in mean_score_log_v]
    else:        
        if w_function == 'inv_sum' or w_function == None:
            weights = [1 / score for score in mean_score_valid]
        elif w_function == 'inv_sq_sum':
            weights = [1 / score**2 for score in mean_score_valid]
    weights = np.array(weights) / np.sum(weights)
    full_data_score = rmsle(y, weighted_average(y_full_pred, weights))
    for n in range(n_models):
        print 'Model', n
        print 'Mean training score (RMSLE): {0:.5f}'.format(mean_score_train[n])
        print 'Mean validation score (RMSLE): {0:.5f}'.format(mean_score_valid[n])
        if y_type == 'log1p':
            print 'Mean training score (RMSLE) for correct (exp) data: {0:.5f}'.format(mean_score_log_t[n])
            print 'Mean validation score (RMSLE) for correct (exp) data: {0:.5f}'.format(mean_score_log_v[n])
    print ' '
    print 'Mean weighted training score (RMSLE): {0:.5f}'.format(mean_weighted_score_train)
    print 'Mean weighted validation score (RMSLE): {0:.5f} <--'.format(mean_weighted_score_valid)
    if y_type == 'log1p':
        print 'Mean weighted training score (RMSLE) for correct (exp) data: {0:.5f}'.format(mean_score_log_train)
        print 'Mean weighted validation score (RMSLE) for correct (exp) data: {0:.5f} <--'.format(mean_score_log_valid)
    print 'Final weights:', weights
    print ' '
    print 'Full data training score (RMSLE): {0:.5F}'.format(full_data_score)
    if y_type == 'log1p':
        score = rmse(y, weighted_average(y_full_pred, weights))
        print 'Full data training score (RMSLE) for correct (exp) data: {0:.5F}'.format(score)
    print ''
    if y_type != 'log1p':
        return models, weights, mean_weighted_score_valid
    elif y_type == 'log1p':
        return models, weights, mean_score_log_valid

def make_model(X, Y, model_def, params, cv=5, y_type=None, grid_type=None, n_iter=None, n_jobs=1):
    if y_type == None:
        rmsle_scorer = make_scorer(rmsle, greater_is_better=False)
    elif y_type == 'log1p':
        rmsle_scorer = make_scorer(rmse, greater_is_better=False)
    if grid_type == None or grid_type == 'grid':
        model = GridSearchCV(model_def, params, scoring=rmsle_scorer, cv=cv, verbose=1, n_jobs=n_jobs).fit(X, Y)
    elif grid_type == 'random':
        model = RandomizedSearchCV(model_def, params, n_iter=n_iter, scoring=rmsle_scorer, cv=cv, verbose=1, n_jobs=n_jobs).fit(X, Y)
    for score in model.grid_scores_:
        print score
    print ' '
    print 'Best valid score (RMSLE): ', -model.best_score_
    if y_type == None:
        print 'Full data training score: ', rmsle(model.predict(X), Y)
    elif y_type == 'log1p':
        print 'Full data training score: ', rmse(model.predict(X), Y)
    print model.best_params_
    print ''
    return model
    
def fit_submit(X_test, test_id, model, filename, price_type=None):
    print 'Fitting model on test data...'
    prices = model.predict(X_test)
    if price_type == 'log1p':
        prices = np.exp(prices) - 1
    submission = pd.DataFrame({'Id': test_id, 'SalePrice': prices})
    submission.to_csv(filename, index=False)
    print 'Done'
    print ''
    
def fit_submit_array(X_test, test_id, models, weights, filename, price_type=None):
    print 'Fitting model on test data...'
    y_preds = []
    for model in models:
        if price_type == None:
            y_preds.append(model.predict(X_test))
        elif price_type == 'log1p':
            y_preds.append(np.exp(model.predict(X_test)) - 1)
    prices = weighted_average(y_preds, weights)
    submission = pd.DataFrame({'Id': test_id, 'SalePrice': prices})
    submission.to_csv(filename, index=False)
    print 'Done'
    print ''