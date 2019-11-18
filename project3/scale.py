# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:57:32 2019

@author: made_
"""

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def scale(X_train, X_test):
    scaler = StandardScaler().fit(X_train)

    # scale
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled
