from sklearn.ensemble import RandomForestClassifier
from run import run

PROTOTYPING = True
NUMBER_ESTIMATORS = 10

run(classifier=RandomForestClassifier(n_estimators=NUMBER_ESTIMATORS), proto=PROTOTYPING)
