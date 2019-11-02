import pandas as pd
from sklearn.utils import resample

def sampling(X_train, y_train, X_columns, y_columns):
    ## Sampling
    X_train = pd.DataFrame(data=X_train, columns=X_columns)
    y_train = pd.DataFrame(data=y_train, columns=y_columns)

    X = pd.concat([y_train, X_train], axis=1)

    # TODO: make generic instead of hardcoded
    label_0 = X[X.y==0]
    label_1 = X[X.y==1]
    label_2 = X[X.y==2]

    label_0_upsampled = resample(label_0, n_samples=len(label_1), random_state=0)
    label_2_upsampled = resample(label_2, n_samples=len(label_1), random_state=0)

    upsampled = pd.concat([label_0_upsampled, label_1, label_2_upsampled]).sample(frac=1, random_state=0)

    y_train = upsampled.values[:, 0]
    X_train = upsampled.values[:, 1:]
    return X_train, y_train
