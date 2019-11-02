from sklearn.svm import NuSVC
from run import run

PROTOTYPING = False

run(classifier=NuSVC(nu=0.24, gamma='scale'), proto=PROTOTYPING)
