import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from scale import scale
from sampling import sampling
from preprocessing import preprocess

def run(classifier, proto = False):

    ## Reading input
    X_train = pd.read_csv('files/X_train.csv').drop('id', axis=1)
    X_test = pd.read_csv('files/X_test.csv').drop('id', axis=1)
    y_train = pd.read_csv('files/y_train.csv').drop('id', axis=1)

    X_columns = X_train.columns.values
    y_columns = y_train.columns.values

    ## Feature scaling
    X_train, X_test = scale(X_train, X_test)

    ## Splitting for validation
    if proto:
        X_train, X_test, y_train, y_test = preprocess(X_train, y_train)
        print("Finished splitting");

    X_train, y_train = sampling(X_train, y_train, X_columns, y_columns)
    print("Finished sampling");

    ## Trainining
    OvRClassifier = OneVsRestClassifier(classifier)
    OvRClassifier.fit(X_train, y_train)
    print("Finished training");

    ## Predicting
    y_predict = OvRClassifier.predict(X_test)
    if proto:
        print(balanced_accuracy_score(y_test, y_predict))

    ## Writing output
    if not proto:
        output = pd.read_csv('files/sample.csv')
        for i in range(output.shape[0]):
            output.iat[i, 1] = y_predict[i]
        output.to_csv(f"outputs/{OvRClassifier.__class__.__name__}.{classifier.__class__.__name__}.csv", index=False)
    print("Finished predicting");

