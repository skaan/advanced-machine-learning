import pandas as pd
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE, ADASYN, KMeansSMOTE

ALGORITHM = "smote" # { oversampling, undersampling, smote, adasyn } 

# TODO: make generic instead of hardcoded
def sampling(X_train, y_train, X_columns, y_columns):

    # Format data
    X_train = pd.DataFrame(data=X_train, columns=X_columns)
    y_train = pd.DataFrame(data=y_train, columns=y_columns)
    X = pd.concat([y_train, X_train], axis=1)

    label_0 = X[X.y==0]
    label_1 = X[X.y==1]
    label_2 = X[X.y==2]

    if ALGORITHM == "oversampling":
        label_0_upsampled = resample(label_0, n_samples=len(label_1), random_state=0)
        label_2_upsampled = resample(label_2, n_samples=len(label_1), random_state=0)
        upsampled = pd.concat([label_0_upsampled, label_1, label_2_upsampled]).sample(frac=1, random_state=0)
        y_train = upsampled.values[:, 0].astype(int)
        X_train = upsampled.values[:, 1:].astype(int)

    elif ALGORITHM == "undersampling":
        label_1_downsampled = resample(label_1, n_samples=len(label_0), random_state=0)
        downsampled = pd.concat([label_0, label_1_downsampled, label_2]).sample(frac=1, random_state=0)
        y_train = downsampled.values[:, 0].astype(int)
        X_train = downsampled.values[:, 1:].astype(int)

    elif ALGORITHM == "smote":
        sm = SMOTE(sampling_strategy="auto", k_neighbors=10)
        X_train, y_train = sm.fit_sample(X_train, y_train)

    elif ALGORITHM == "adasyn":
        sm = SMOTE(sampling_strategy="auto")
        X_train, y_train = sm.fit_sample(X_train, y_train)
    return X_train, y_train
