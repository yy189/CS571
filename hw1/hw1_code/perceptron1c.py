import numpy as np
from sklearn.datasets import fetch_mldata
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt

mnist_dataset = fetch_mldata('MNIST original')
X_4 = mnist_dataset['data'][np.where(mnist_dataset['target'] == 4.)[0]] / 255
X_9 = mnist_dataset['data'][np.where(mnist_dataset['target'] == 9.)[0]] / 255
Y_4 = np.full((X_4.shape[0], 1), 1)
Y_9 = np.full((X_9.shape[0], 1), -1)
X = np.vstack((X_4, X_9))#concatenate
Y = np.vstack((Y_4, Y_9))
permutation = np.random.RandomState(seed=2018).permutation(Y.shape[0])#shuffle dataset and labels
X = X[permutation, :]
Y = Y[permutation]
X_train, X_test = X[:8270], X[8270:]#60% trainig, 40% testing
Y_train, Y_test = Y[:8270], Y[8270:]

def perceptron(X, Y):
    w = np.zeros(len(X[0]))
    epochs = 100

    for e in range(epochs):
        for i, x in enumerate(X):
            if(np.dot(X[i], w)*Y[i]) <= 0:
                w = w + X[i]*Y[i]

    return w

w = perceptron(X_train, Y_train)

TP = FP = FN = TN = 0

for i, x in enumerate(X_test):
    if np.dot(X_test[i], w) > 0:
        if Y_test[i] == 1:
            TP += 1
        else:
            FP += 1
    else:
        if Y_test[i] == 1:
            FN += 1
        else:
            TN += 1

print('TP: ' + str(TP) + ', FP: ' + str(FP) + ', FN: ' + str(FN) + ' TN: ' + str(TN) + ', Accuracy: ' + str((TP+TN)/5512))

