#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 12:05:34 2017

@author: ratnadeepb
"""

import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

############# Exit on erronous input #################
def exit_err(ip, message):
    """
    This is called when the input to process function is not correct
    @Output:
    """
    print("Incorrect input: ", message, " ", ip)
    sys.exit(-1)

def process(filename, cat, numerical_only, response, add_bias=True,
            test_size=0.25, **kwargs):
    """
    Preprocess the data mostly autonomously
    @Inputs:
        filename: file name that holds the data
        cat: If there are categorical features
        numerical_only: If scaling is to be done on numerical data only
        add_bias: If True add the bias column
        response: The response variable column name
        test_size: The size of the test data set
        kwargs - cols: The categorical variable columns.
    @Output:
        [X_train, X_test, y_train, y_test, columns,
        X_mean, X_std, y_mean, y_std]
    """

    if not isinstance(cat, bool): exit_err(cat, 'Boolean expected for cat')
    if not isinstance(numerical_only, bool): exit_err(numerical_only,
                     'Boolean expected for numerical_only')
    if not (0.0 < test_size and test_size < 1.0): exit_err(test_size,
           'Wrong value for test_size: expected above 0 and below 1')

    # Check that response is a list
    if not isinstance(response, list):
        r = []
        r.append(response)
        response = r

    # If kwargs exists then cat should be True
    # If kwargs exists then it should have just one key - cols
    # cols should be a list of categorical variables
    if kwargs:
        if not cat:
            exit_err(cat, 'Are there categorical variables?')
        elif 'cols' not in kwargs.keys():
            exit_err(kwargs, 'keyword missing in kwargs')
        else:
            if not isinstance(kwargs['cols'], list):
                exit_err(kwargs['cols'], 'Not a list')

    #################### Import Data ####################
    try:
        data = pd.read_csv(filename)
    except FileNotFoundError:
        exit_err(filename, "Incorrect Filename")

    columns = data.columns  # Reference to feature names

    ############### Deal with Missing Values ############
    ### This is simplistic
    if cat:
        for col in columns:
            missing = tuple(data[col].isnull()[data[col].isnull()==True].index)
            if missing:
                if col in kwargs['cols']:
                    data[col].iloc[missing] = data[col].mode()[0]
                else:
                    data[col].iloc[missing] = data[col].mean()

    ############### Encoding Categorical Variables ############
    if cat:
        try:
            cols = kwargs['cols']
        except (KeyError|NameError) as err:
            exit_err('cat', err)
        data = pd.get_dummies(data, columns=cols, drop_first=True)

    #################### Add Bias ####################
    if add_bias:
        data.insert(loc=0, column='Bias', value=np.ones((data.shape[0], 1)))

    #################### Separate into features and target ####################
    columns = data.columns  # Column names have now changed
    ind_vars = [col for col in columns if col not in response]
    try:
        X = data[ind_vars]
    except KeyError as e:
        exit_err(ind_vars, e)
    try:
        y = data[response]
    except KeyError as e:
        exit_err(response, e)

    #################### Label Encoding Response ####################
    y_labels = LabelEncoder()
    y = pd.DataFrame(y_labels.fit_transform(y))

    #################### Feature Scaling ####################
    to_be_used = [col for col in ind_vars if col != 'Bias']

    y_mean = y.mean()
    y_std = y.std()
    y = (y - y_mean) / y_std

    if not numerical_only:
        X_mean = X[to_be_used].mean()
        X_std = X[to_be_used].std()

        X[to_be_used] = (X[to_be_used] - X_mean) / X_std

    else:
        numericals = []
        for c in cols:
            for t in to_be_used:
                if c not in t:
                    numericals.append(t)

        X_mean = X[numericals].mean()
        X_std = X[numericals].std()

        X[numericals] = (X[numericals] - X_mean) / X_std

    #################### Train -Test Set ####################
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size)

    return [X_train, X_test, y_train, y_test, columns,
            X_mean, X_std, y_mean, y_std]