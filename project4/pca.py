# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 09:06:21 2019

@author: made_
"""
# PCA on EEG datasets
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA

# import eeg features directly
eeg_train = pd.read_csv('files/eeg_feats_train.csv').values
eeg_test  = pd.read_csv('files/eeg_feats_test.csv').values
y_train   = pd.read_csv('files/train_labels.csv').drop('Id', axis=1).values

# scale
scaler = StandardScaler()
scaler.fit(eeg_train)
eeg_train_s = scaler.transform(eeg_train)
eeg_test_s  = scaler.transform(eeg_test)

# perfrom pca
pca = PCA(n_components=45, random_state=0)
pca.fit_transform(eeg_train_s, y_train)
#print('Explained Variance: %0.5f' % np.sum(pca.explained_variance_ratio_[0:45]))  # looks like 45 are enough

# transform scaled eeg datasets
eeg_train_mod = pca.transform(eeg_train_s)
eeg_test_mod  = pca.transform(eeg_test_s)

# write
pd.DataFrame.to_csv(pd.DataFrame(eeg_train_mod), 'files/eeg_train_pca45.csv', index=False)
pd.DataFrame.to_csv(pd.DataFrame(eeg_test_mod), 'files/eeg_test_pca45.csv', index=False)
