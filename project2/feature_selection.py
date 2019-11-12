import numpy as np
import pandas as pd
from scale import scale
from sampling import sampling
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

NUMBER_FOLDS = 2

# Reading input
X_train = pd.read_csv('files/X_train.csv').drop('id', axis=1)
X_test = pd.read_csv('files/X_test.csv').drop('id', axis=1)
y_train = pd.read_csv('files/y_train.csv').drop('id', axis=1)

X_columns = X_train.columns.values
y_columns = y_train.columns.values

## Feature scaling
X_train, X_test = scale(X_train, X_test)

## Sampling
X_train, y_train = sampling(X_train, y_train, X_columns, y_columns)

print(X_train.shape, X_test.shape)

estimator = SVC(kernel="linear", gamma='scale')

rfecv = RFECV(estimator, step=1, cv=StratifiedKFold(NUMBER_FOLDS), scoring='accuracy', verbose=1)
rfecv.fit(X_train, y_train)

print("Optimal number of features : %d" % rfecv.n_features_)

X_train_fs = rfecv.transform(X_train)
X_test_fs = rfecv.transform(X_test)

print(X_train_fs.shape, X_test_fs.shape)

np.savetxt(f"files/X_train_fs_{NUMBER_FOLDS}.csv", X_train_fs, delimiter=",")
np.savetxt(f"files/X_test_fs_{NUMBER_FOLDS}.CSV", X_test_fs, delimiter=",")
