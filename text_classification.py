import numpy as np
from sklearn.model_selection import train_test_split
from torch import nn
import sister

from skorch import NeuralNetClassifier

train_texts = [
        'heppy new year',
        'well done',
        'nice to meet you',
        'hate you',
        'go to hell',
        'get lost'
        ]
test_texts = [
        'i love you',
        'welcome back',
        'go away',
        'who cares'
        ]
train_y = np.array([0, 0, 0, 1, 1, 1]).astype(np.int64)
test_y = np.array([0, 0, 1, 1]).astype(np.int64)


class MyModule(nn.Module):

    def __init__(self, num_units=10, nonlin=nn.ReLU()):
        super().__init__()

        self.dense0 = nn.Linear(300, num_units)
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
        iterator_train__shuffle=True,
        train_split=None
        )

sentence_embedding = sister.MeanEmbedding(lang="en")
train_x = np.array([sentence_embedding(text) for text in train_texts]).astype(np.float32)
test_x = np.array([sentence_embedding(text) for text in test_texts]).astype(np.float32)

net.fit(train_x, train_y)
print(net.score(train_x, train_y))
print(net.score(test_x, test_y))
