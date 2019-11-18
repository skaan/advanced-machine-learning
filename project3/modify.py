# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:53:27 2019

@author: made_
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from scale import scale
from outliers import outliers
from impute import impute

# TODO : give it suitable arguments based on run.py
def modify():
    # import data
    X_train = pd.read_csv('files/X_train.csv').values[:,1:]
    y_train = pd.read_csv('files/y_train.csv').values[:,1 ]
    X_test  = pd.read_csv('files/X_test.csv').values[:,1:]
    
    summary = []  # final summary output
    z,s = np.shape(X_train)
    z_,s_ = np.shape(X_test)
    
    # check if balanced
    print('Size of X:', np.shape(X_train))
    print('Classes in y:',np.unique(y_train))
    summary.append('Classes in y: %s' %np.unique(y_train))
    iter_i = 0
    print('Class Balance')
    while iter_i != len(np.unique(y_train)):
        print('    Data for Class %s: ' %iter_i, len(y_train[np.where(y_train==iter_i)]))
#        summary.append('    Data for Class %s: ' %iter_i, len(y_train[np.where(y_train==iter_i)], '\n'))
        iter_i += 1
    
    
    # check for presence of nans
    isnan_train = np.zeros(s, dtype=int)
    isnan_test  = np.zeros(s, dtype=int)
    for i in range(s):
        isnan_train[i] = np.sum(np.isnan(X_train[:,i]))
        isnan_test[i] = np.sum(np.isnan(X_test[:,i]))
        
#    print(list(isnan_train))
#    print(len(list(isnan_train[isnan_train < 1000])))
#    print(list(isnan_test))
#    print(len(list(isnan_test[isnan_test < 1000])))
    
    # modify to keep only those features with less than 1000 nans in X_train
    X_train_mod = X_train[:,isnan_train < 1000]
    X_test_mod  = X_test[:,isnan_train < 1000]  # apply same reduction to testset as in trainset
    
#    summary.append('Reduced X_train from shape (%s,%s) to (%s,%s)' %(z,s,np.shape(X_train_mod)))
#    summary.append('Reduced X_test from shape (%s,%s) to (%s,%s)' %(z_,s_,np.shape(X_test_mod)))
    
    
    # use simple procedure to replace nans
    X_train_mod = impute(X_train_mod)
    X_test_mod  = impute(X_test_mod)
    
    
    # check for outliers
    train_inliers = outliers(X_train_mod)
    test_inliers = outliers(X_test_mod)
    X_train_mod = X_train_mod[train_inliers, :]
    X_test_mod = X_test_mod[test_inliers, :]
    y_train = y_train[train_inliers]
    
    
    # scale data after the previous procedure
    X_train_mod, X_test_mod = scale(X_train_mod, X_test_mod)
    
    # TODO : add automatic saving of modified data for quicker usage
    
    return X_train_mod, y_train, X_test_mod  # return also summary