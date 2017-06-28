#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"Created on 24 Jun 21:30 2017"
"@author: ratnadeepb"

import pandas as pd
import numpy as np
from scipy.stats import mode
from itertools import product
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#################### Import Data ####################
data = pd.read_csv("/home/ratnadeepb/app/machine_learning/ml_programming/Data/Data.csv")
# data = pd.read_csv("Data/Data.csv")
columns = data.columns  # Reference to feature names

#################### Separate into features and target ####################
X = data[['Country', 'Age', 'Salary']].values
y = data['Purchased'].values


############### Separating Categorical and Numerical Data ############
cat_cls = [0]  # Row numbers in X with categorical data
cls = [cl for cl in range(X.shape[1]) if cl not in cat_cls]

C = np.array([list(elem) for elem in zip(*[X[:, i] for i in cat_cls])])
C = C.reshape(C.shape[0], -1)
N = np.array([list(elem) for elem in zip(*[X[:, i] for i in cls])])
N = N.reshape(N.shape[0], -1)

#################### Handling missing numerical data ####################
# Impute numerical variables
imputer_num = Imputer()
N = imputer_num.fit_transform(N)
c_cls = {}
for cl in cat_cls:
    c_cls[cl] = cat_cls.index(cl)
vals = {cl:mode(C[:,cl])[0][0] for cl in c_cls.values()}

missing_indices = [[rw,cl] for rw, cl in product(range(C.shape[0]), range(C.shape[1]))
                   if pd.isnull(C[rw, cl]) or  C[rw, cl] == 'nan']

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
                   if pd.isnull(y[ind]) or  y[ind] == 'nan']

for i in missing_y:
    y[i] = response

y_labels = LabelEncoder()
y = y_labels.fit_transform(y)

#################### Train -Test Set ####################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#################### Feature Scaling ####################
# Here we are scaling the categorical variables
# The scaling can also be done to the numerical variables only
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_train)