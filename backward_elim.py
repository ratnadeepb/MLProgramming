#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"Created on 30 Jun 17:20 2017"
"@author: ratnadeepb"

env = "my_code"

import sys
if env not in sys.path:
    sys.path.append(env)
from process_v1 import process
from process_v1 import column_prep
from model_score import adjusted_r_squared
from sklearn.externals import joblib
from sklearn import metrics
import statsmodels.formula.api as sm
import numpy as np

def backward_elim(file, save_to, sl_stay, rows):
    #################### Prepare the file ####################
    if len(rows) == 0:
        X_train, X_test, y_train, y_test, columns, original_tags = process(file,
                                                                       False,
                                                                       True, True, 1/3)
    else:
        X_train, X_test, y_train, y_test, columns, original_tags = process(file,
                                                                           True,
                                                                           True, True, 1 / 3, rows=rows)

    #################### Prepare the feature list ####################
    features = column_prep(rows, columns, original_tags)

    #################### Building the regressor ####################
    X_opt = X_train[:,:]
    regressor_OLS = sm.OLS(y_train, X_opt).fit()
    pvals = regressor_OLS.pvalues
    max_pval = pvals.max()
    while max_pval > sl_stay:
        index = np.where(pvals==max_pval)[0][0]
        X_opt = np.delete(X_opt, index, 1)
        X_test = np.delete(X_test, index, 1)
        features.pop(index)
        regressor_OLS = sm.OLS(y_train, X_opt).fit()
        pvals = regressor_OLS.pvalues
        max_pval = pvals.max()

    #################### Saving the model ####################
    joblib.dump(regressor_OLS, save_to)

    #################### Predicting ####################
    predictions = regressor_OLS.predict(X_test)

    #################### Adjusted R-Squared ####################
    r2_score = metrics.r2_score(y_test, predictions)
    adj_r2_score = adjusted_r_squared(X_test.shape[0], X_test.shape[1], r2_score)

    return [regressor_OLS.params, adj_r2_score, features]

if __name__ == "__main__":
    rows = [3]
    sl_stay = 0.05
    file = "/home/ratnadeepb/app/machine_learning/ml_programming/Data/50_Startups.csv"
    save_to = "/home/ratnadeepb/app/machine_learning/ml_programming/saved_models/backward_elim.pkl"
    l, adj_r2, features = backward_elim(file, save_to, 0.05, rows)
    print("The coefficient values are: ", l)
    print("The accurace of the model is: {}%".format(round(adj_r2, 2) * 100))
    print("The final feature list is: ", features)