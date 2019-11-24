# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 15:26:55 2019

@author: made_
"""

import numpy as np

# perturb data by adding a bit of random Gaussian noise for every feature
def perturb(X, mean=0, std=1):
    z, s = np.shape(X)
    for i in range(s):
        noise = np.random.normal(mean, std, z)  # create random noise for i-th feature
        X[:,i] += noise
        
    return X