from sklearn.impute import SimpleImputer
import numpy as np

def impute(X, strategy = 'mean'):
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean.fit(X)
    X_mod = imp_mean.transform(X)
    return X_mod