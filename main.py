import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch import nn

from skorch import NeuralNetClassifier

dataset = load_iris()
X, y = dataset.data, dataset.target
train_x, test_x, train_y, test_y = train_test_split(X, y)
train_x = train_x.astype(np.float32)
test_x = test_x.astype(np.float32)
train_y = train_y.astype(np.int64)
test_y = test_y.astype(np.int64)


class MyModule(nn.Module):

    def __init__(self, num_units=10, nonlin=nn.ReLU()):
        super().__init__()

        self.dense0 = nn.Linear(4, num_units)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(0.5)
        self.dense1 = nn.Linear(num_units, num_units)
        self.output = nn.Linear(num_units, 3)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = self.nonlin(self.dense1(X))
        X = self.softmax(self.output(X))
        return X


net = NeuralNetClassifier(
        MyModule,
        max_epochs=300,
        lr=0.1,
        iterator_train__shuffle=True
        )

net.fit(train_x, train_y)
print(net.score(train_x, train_y))
print(net.score(test_x, test_y))
