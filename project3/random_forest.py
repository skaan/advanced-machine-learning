# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 17:10:53 2019

@author: made_
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.utils import compute_class_weight
from run import run

# get data
#X_train = pd.read_csv('files/X_train_aug.csv').values[:,1:]
#y_train = pd.read_csv('files/y_train_aug.csv').values[:,1 ]
#X_test  = pd.read_csv('files/X_test_aug.csv').values[:,1:]

PROTOTYPING = True

# do a parameter search
n_trees = np.array([40,50,60,70,80])
boot_opt = [False, True]
random_state = 1000
weights = [None, 'balanced', 'balanced_subsample']

param_grid = [{'n_estimators' : n_trees,
               'bootstrap' : boot_opt,
               'oob_score' : boot_opt,
               'class_weight' : [None, 'balanced', 'balanced_subsample']}]

# run ExtraTreesClassifier and RandomForestClassifier separately
for i in range(len(n_trees)):
    for j in range(len(boot_opt)):
        for k in range(len(weights)):
            print('Estimators =', n_trees[i])
            print('Bootstrap =', boot_opt[j])
            print('Weights are ', weights[k])
            run(classifier=RandomForestClassifier(n_estimators=n_trees[i],
                                                bootstrap=boot_opt[j], oob_score=boot_opt[j],
                                                class_weight=weights[k],
                                                random_state=random_state), proto=PROTOTYPING)
            print('')
            
## GridSearchCV
#clf = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring = 'accuracy')
#for i in range(len(n_trees)):
#    print('Estimators =', n_trees[i])
#    clf.fit(X_train, y_train)
#    print('')
#print(clf.best_params_)


####
# Do the final prediction
PROTOTYPING = False
# those are the best params from the parameter search
run(classifier=RandomForestClassifier(n_estimators=100, bootstrap=True, oob_score=True,
                                      class_weight='balanced_subsample'), proto=PROTOTYPING)
