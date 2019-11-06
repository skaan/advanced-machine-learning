# Outlier detection

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# testing site
#X_train = pd.read_csv('files/X_train.csv')
#X_train = X_train.values[:,1:]
#Y_train = pd.read_csv('files/y_train.csv').values[:,1]
#n_trees = 50

# perform outlier detection
def find_outlier(X_train, Y_train = [0], n_trees = 50):
    print(Y_train)
    outlier = IsolationForest(n_estimators=n_trees,
                              behaviour='new',
                              contamination=0.1)
    outlier.fit(X_train)
    inlier_indices = np.nonzero(outlier.predict(X_train) > 0)[0]
    X_train = X_train[inlier_indices, :]
    if len(Y_train) != 1:
        Y_train = Y_train[inlier_indices]
        return X_train, Y_train
    else:
        return X_train