import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from modify import modify
from augment import augment
from scale import scale
from sampling import sampling
from preprocess import preprocess


# process data: creates modified train and testsets for faster import and use
# modify()   # creates modified dataset
# augment()  # creates augmented dataset


# for unbalanced data: OvO, OvR or None
def run(classifier, proto = False):

    ## Reading input
    X_train = pd.read_csv('files/X_train_aug.csv').drop('Unnamed: 0', axis=1)
    X_test = pd.read_csv('files/X_test_aug.csv').drop('Unnamed: 0', axis=1)
    y_train = pd.read_csv('files/y_train.csv').drop('id', axis=1)

    X_columns = X_train.columns.values
    y_columns = y_train.columns.values
    

    ## Splitting for validation
    if proto:
        X_train, X_test, y_train, y_test = preprocess(X_train, y_train)
        print("Finished splitting");
    
    # sampling adds one row more to X_train for some reason... also make generic
    # this part is probably not needed for rfcs as it does it itself sometimes
    X_train, y_train = sampling(X_train, y_train, X_columns, y_columns)
    print("Finished sampling");
    
    print(np.shape(X_train))
    print(np.shape(X_test))
    print(np.shape(y_train))

    ## Trainining
    OvRClassifier = OneVsOneClassifier(classifier)
    OvRClassifier.fit(X_train, y_train)
    print("Finished training");

    ## Predicting
    y_predict = OvRClassifier.predict(X_test)
    if proto:
        print(f1_score(y_test, y_predict, average='weighted'))

    ## Writing output
    if not proto:
        output = pd.read_csv('files/sample.csv')
        for i in range(output.shape[0]):
            output.iat[i, 1] = y_predict[i]
        output.to_csv(f"outputs/{OvRClassifier.__class__.__name__}.{classifier.__class__.__name__}.csv", index=False)
    print("Finished predicting");

