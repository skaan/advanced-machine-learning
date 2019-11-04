
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from sklearn.metrics import balanced_accuracy_score
from scale import scale
from sampling import sampling
from split import split
from reshape_class_vector import reshape_class_vector

PROTO = True
LEARNING_RATE = 0.00001
NUM_ITERS = 10000
OPTIMIZER_CONSTRUCTOR = optim.Adam

## Reading input
X_train = pd.read_csv('files/X_train.csv').drop('id', axis=1)
X_test = pd.read_csv('files/X_test.csv').drop('id', axis=1)
y_train = pd.read_csv('files/y_train.csv').drop('id', axis=1)

X_columns = X_train.columns.values
y_columns = y_train.columns.values

## Feature scaling
X_train, X_test = scale(X_train, X_test)

## Splitting for validation
if PROTO:
    X_train, X_test, y_train, y_test = split(X_train, y_train)
    print("Finished splitting");

## Sampling
X_train, y_train = sampling(X_train, y_train, X_columns, y_columns)
y_test = y_test.values.reshape(y_test.shape[0])
print("Finished sampling");

# Helper variables
input_feature_size = X_train.shape[1]
hidden_layer_0_size = 800
hidden_layer_1_size = 300
output_feature_size = 3

## Trainining
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(input_size, input_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(input_size, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return out

model = Net(input_feature_size, hidden_layer_0_size, output_feature_size)

#Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Reshape from vector with labels to matrix with column for each class
y_train = reshape_class_vector(y_train)
y_test_reshaped = reshape_class_vector(y_test)

# Formatting to Tensor
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()

y_test_reshaped = torch.from_numpy(y_test_reshaped).float()

for epoch in range(NUM_ITERS):
    # zero the parameter gradients

    optimizer.zero_grad()
    y_predict = model(X_train)

    current_loss = criterion(y_predict, y_train)
    current_loss.backward()
    optimizer.step()

    y_predict = np.argmax(model(X_test).detach().numpy(), axis=1)
    print(epoch, balanced_accuracy_score(y_test, y_predict))

print('Finished Training')

## Writing output
if not PROTO:
    output = pd.read_csv('files/sample.csv')
    for i in range(output.shape[0]):
        output.iat[i, 1] = y_predict[i]
    output.to_csv(f"outputs/deep_neural_network.csv", index=False)
print("Finished predicting");
