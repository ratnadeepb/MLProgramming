#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"Created on 30 Jun 10:45 2017"
"@author: ratnadeepb"

env = "my_code"

import sys
if env not in sys.path:
    sys.path.append(env)
from process_v1 import process
from model_score import adjusted_r_squared
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.externals import joblib

def mul_lin_reg(file, save_to, rows):
    """
    Multiple Linear Regression
    :param file: File containing the data
    :param save_to: File to which the model will be saved
    :param rows: rows that are categorical
    :return: The adjusted R-squared score and the coefficients of the model
    """

    #################### Prepare the file ####################
    if len(rows) == 0:
        X_train, X_test, y_train, y_test, columns, original_tags = process(file,
                                                                       False,
                                                                       True, True, 1/3)
    else:
        X_train, X_test, y_train, y_test, columns, original_tags = process(file,
                                                                           True,
                                                                           True, True, 1 / 3, rows=rows)
    #################### Building the regressor ####################
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    #################### Saving the model ####################
    joblib.dump(regressor, save_to)

    #################### Predicting ####################
    predictions = regressor.predict(X_test)

    #################### Adjusted R-Squared ####################
    r2_score = metrics.r2_score(y_test, predictions)
    adj_r2_score = adjusted_r_squared(X_test.shape[0], X_test.shape[1], r2_score)

    coeffs = [regressor.intercept_]
    coeffs.extend(regressor.coef_)

    return [coeffs, adj_r2_score]

if __name__ == "__main__":
    rows = [3]
    file = "/home/ratnadeepb/app/machine_learning/ml_programming/Data/50_Startups.csv"
    save_to = "/home/ratnadeepb/app/machine_learning/ml_programming/saved_models/mul_lin_reg.pkl"
    l, adj_r2 = mul_lin_reg(file, save_to, rows)

    l2 = []
    for i in l:
        l2.append(round(i, 3))
    print("The coefficient values are: ", l2)
    print("The accurace of the model is: {}%".format(round(adj_r2, 4) * 100))