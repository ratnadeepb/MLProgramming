#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"Created on 27 Jun 18:01 2017"
"@author: ratnadeepb"

def adjusted_r_squared(n, d, r2_score):
    """
    Returns the Adjusted r-squared
    :param n: Number of data points
    :param d: Number of dimensions in the data
    :param r2_score: r2_score
    :return: Adjusted r2_score
    """
    return 1 - (((1 - r2_score) * (n - 1)) / (n - d - 1))