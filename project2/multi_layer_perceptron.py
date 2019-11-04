from sklearn.neural_network import MLPClassifier
from run import run

PROTOTYPING = True

run(classifier=MLPClassifier(hidden_layer_sizes=(1000, 300), learning_rate="adaptive"), multi=False, proto=PROTOTYPING)
