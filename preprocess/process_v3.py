#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 13:45:51 2018

@Author: RatnadeepB
@License: MIT
"""

import sys
import numpy as np
import pandas as pd
# needed for categorical variables
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import re

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
        [X_train, X_test, y_train, y_test, columns]
        
    @Details:
        If kwargs exists then cat should be True
        If kwargs exists then it should have just one key - cols
        cols should be a list of categorical variables
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
            missing = tuple(data[data[col].isnull() == True].index)
            if missing:
                if col in kwargs['cols']: # this is a categorical variable
                    # Replace missing values in categorical variables with mode
                    data.loc[missing, col] = data[col].mode()[0]
                else:
                    # Replace missing values in non-categorical variables with mean
                    data.loc[missing, col] = data[col].mean()

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
    y = y.apply(LabelEncoder().fit_transform)
    
    #################### Change to numpy arrays ####################
    X = X.values
    y = y.values
    
    X = X.astype(np.float64)
    y = y.astype(np.float64)
    
    #################### Train -Test Set ####################
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size)

    #################### Feature Scaling ####################
    to_be_used = [col for col in ind_vars if col != 'Bias']

    sc_y = StandardScaler()
    y_train = sc_y.fit_transform(y_train)
    y_test = sc_y.transform(y_test)
    
    sc_X = StandardScaler()
    if not numerical_only:
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(X_test)
    else:
        numericals = []
        add_col = True
        for t in to_be_used:
            for c in cols:
                if re.match(c, t): # for all c in cols, check if c == t
                    add_col = False # t is a categorical variable                    
            if add_col:
                numericals.append(t)
        nums = []
        for i, c in enumerate(columns):
            for n in numericals:
                if n == c:
                    nums.append(i)
        X_train[nums] = sc_X.fit_transform(X_train[nums])
        X_test[nums] = sc_X.transform(X_test[nums])

    return [X_train, X_test, y_train, y_test, columns]