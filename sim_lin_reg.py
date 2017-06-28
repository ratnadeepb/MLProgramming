#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"Created on 27 Jun 16:05 2017"
"@author: ratnadeepb"
env = "my_code"

import sys
if env not in sys.path:
    sys.path.append(env)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn import metrics
from process_v1 import process
from model_score import adjusted_r_squared

def sim_lin_reg(file, splits):
    """
    Simple Linear Regression
    :param file: File containing the data
    :param file: Splits for K-Fold validation
    :return: The adjusted R-squared score
    """

    #################### Prepare the file ####################
    X_train, X_test, y_train, y_test, columns, original_tags = process(file,
                                                                       False,
                                                                       True, 1/3)

    #################### Cross Validation ####################

    #################### Building the regressor ####################
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    #################### Predicting ####################
    predictions = regressor.predict(X_test)

    #################### Adjusted R-Squared ####################
    r2_score = metrics.r2_score(y_test, predictions)
    adj_r2_score = adjusted_r_squared(X_test.shape[0], X_test.shape[1], r2_score)

    return [regressor.coef_, adj_r2_score]