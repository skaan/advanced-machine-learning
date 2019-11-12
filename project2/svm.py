from sklearn.svm import NuSVC
from run import run

PROTOTYPING = True

run(classifier=NuSVC(nu=0.5, gamma='scale'), proto=PROTOTYPING)
