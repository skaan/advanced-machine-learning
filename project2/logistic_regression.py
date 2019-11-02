from sklearn.linear_model import LogisticRegression
from run import run

PROTOTYPING = True

run(classifier=LogisticRegression(solver="lbfgs"), proto=PROTOTYPING)

