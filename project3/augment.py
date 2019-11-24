# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 13:30:00 2019

@author: made_
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from scale import scale
from outliers import outliers
from impute import impute
from perturb import perturb


def augment():
    # import data
    X_train = pd.read_csv('files/X_train.csv').values[:,1:]
    y_train = pd.read_csv('files/y_train.csv').values[:,1 ]
    X_test  = pd.read_csv('files/X_test.csv').values[:,1:]
    
    z,s = np.shape(X_train)
    z_,s_ = np.shape(X_test)
    
    # check if balanced
    print('Size of X:', np.shape(X_train))
    print('Classes in y:',np.unique(y_train))
    iter_i = 0
    print('Class Balance')
    while iter_i != len(np.unique(y_train)):
        print('    Data for Class %s: ' %iter_i, len(y_train[np.where(y_train==iter_i)]))
        iter_i += 1
    
    
    # check for presence of nans
    isnan_train = np.zeros(s, dtype=int)
    isnan_test  = np.zeros(s, dtype=int)
    for i in range(s):
        isnan_train[i] = np.sum(np.isnan(X_train[:,i]))
        isnan_test[i] = np.sum(np.isnan(X_test[:,i]))
    
    # modify to keep only those features with less than #n nans in X_train
    X_train_mod = X_train[:,isnan_train < 100]
    X_test_mod  = X_test[:,isnan_train < 100]  # apply same reduction to testset as in trainset
    
    
    # use simple procedure to replace nans
    X_train_mod = impute(X_train_mod)
    X_test_mod  = impute(X_test_mod)
    
    
    # check for outliers - causes problems when writing y_pred to csv
#    train_inliers = outliers(X_train_mod)
#    test_inliers = outliers(X_test_mod)
#    X_train_mod = X_train_mod[train_inliers, :]
#    X_test_mod = X_test_mod[test_inliers, :]
#    y_train = y_train[train_inliers]
    
    
    # scale data after the previous procedure
#    X_train_mod, X_test_mod = scale(X_train_mod, X_test_mod)
    
    
    # data augmentation
    # TODO : augment
    X_train_scaled, X_test_scaled = scale(X_train_mod, X_test_mod)
    X_train_ones = X_train_mod + np.ones_like(X_train_mod)
    X_test_ones  = X_test_mod + np.ones_like(X_test_mod)
    X_train_pert = perturb(X_train_mod)
    X_test_pert  = perturb(X_test_mod)
    X_train_aug  = np.concatenate((X_train_mod, X_train_scaled, X_train_ones, \
                                   X_train_pert), axis=1)
    X_test_aug   = np.concatenate((X_test_mod, X_test_scaled, X_test_ones, \
                                   X_test_pert), axis=1)
    
    # saving of modified data for quicker usage
    pd.DataFrame(X_train_aug).to_csv('files/X_train_aug.csv')
    pd.DataFrame(X_test_aug).to_csv('files/X_test_aug.csv')
#    pd.DataFrame(y_train).to_csv('files/y_train_mod.csv')
    