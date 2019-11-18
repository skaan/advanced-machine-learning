import numpy as np
from sklearn.ensemble import IsolationForest

# remove outliers
def outliers(X, n_trees = 100, outlier_proportion = 0.15):
    outliers = IsolationForest(n_estimators=n_trees,
                               behaviour='new',
                               contamination=outlier_proportion)
    outliers.fit(X)
    inlier_indices = np.nonzero(outliers.predict(X) > 0)[0]
    
    return inlier_indices