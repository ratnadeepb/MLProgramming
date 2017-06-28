#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"Created on 25 Jun 19:24 2017"
"@author: ratnadeepb"

import sys
import pandas as pd
import numpy as np
from scipy.stats import mode
from itertools import product
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def exit_err(this):
    """
    This is called when the input to process function is not correct
    :return:
    """
    print("Incorrect input: ", this)
    sys.exit(-1)


def process(file, cat, numerical_only, test_size, **kwargs):
    """
    Preprocess the data mostly autonomously
    :param file: file name that holds the data
    :param cat: If there are categorical features
    :param numerical_only: If scaling is to be done on numerical data only
    :param test_size: The size of the test data set
    :param kwargs - rows: The categorical variable rows.
    :return: [X_train, X_test, y_train, y_test, columns, original_tags]
    """

    if not isinstance(cat, bool): exit_err(cat)
    if not isinstance(numerical_only, bool): exit_err(numerical_only)
    if not (0.0 < test_size and test_size < 1.0): exit_err(test_size)

    #################### Import Data ####################
    try:
        data = pd.read_csv(file)
    except FileNotFoundError:
        exit_err(file)
    columns = data.columns  # Reference to feature names

    #################### Separate into features and target ####################
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    ############### Separating Categorical and Numerical Data ############
    if cat:
        try:
            cat_cls = kwargs["rows"]  # Row numbers in X with categorical data
        except NameError:
            exit_err("rows")
        cls = [cl for cl in range(X.shape[1]) if cl not in cat_cls]

        C = np.array([list(elem) for elem in zip(*[X[:, i] for i in cat_cls])])
        C = C.reshape(C.shape[0], -1)
        N = np.array([list(elem) for elem in zip(*[X[:, i] for i in cls])])
        N = N.reshape(N.shape[0], -1)

        #################### Handling missing numerical data ####################
        # Impute numerical variables
        imputer_num = Imputer(missing_values='NaN', strategy='mean')
        N = imputer_num.fit_transform(N)
        N_cl_shape = N.shape[1]

        #################### Handling missing categorical data ####################
        c_cls = {}
        for cl in cat_cls:
            c_cls[cl] = cat_cls.index(cl)
        vals = {cl: mode(C[:, cl])[0][0] for cl in c_cls.values()}

        missing_indices = [[rw, cl] for rw, cl in product(range(C.shape[0]), range(C.shape[1]))
                           if pd.isnull(C[rw, cl]) or C[rw, cl] == 'nan']

        for i in missing_indices:
            C[i[0]] = vals[i[1]]

        #################### Encoding categorical data ####################
        original_tags = []
        for cl in list(c_cls.values()):
            original_tags.extend(np.unique(C[:, cl]))
            C_labels = LabelEncoder()
            C = C_labels.fit_transform(C[:, cl])

        C = C.reshape(C.shape[0], -1)

        categorical = []

        for cl in list(c_cls.values()):
            onehotencoder = OneHotEncoder(n_values='auto', categorical_features=[cl])
            temp = onehotencoder.fit_transform(C[:, cl].reshape(-1, 1)).toarray()
            temp = np.delete(temp, -1, 1)  # Dropping the last array
            categorical.extend(temp)

        C = np.array(categorical)

        #################### Recreate features ####################
        X = np.concatenate((C, N), axis=1)

    #################### Handling missing response data ####################
    response = mode(y)[0][0]

    missing_y = [ind for ind in range(y.shape[0])
                 if pd.isnull(y[ind]) or y[ind] == 'nan']

    for i in missing_y:
        y[i] = response

    y_labels = LabelEncoder()
    y = y_labels.fit_transform(y)

    #################### Train -Test Set ####################
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    #################### Feature Scaling ####################
    sc = StandardScaler()
    if cat and numerical_only:
        X_train[:, -N_cl_shape:] = sc.fit_transform(X_train[:, -N_cl_shape:])
        X_test[:, -N_cl_shape:] = sc.transform(X_test[:, -N_cl_shape:])
    else:
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

    if not cat:
        original_tags = []

    return [X_train, X_test, y_train, y_test, columns, original_tags]