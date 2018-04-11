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

def perceptron(X_train, Y_train, X_test, Y_test):
    w = np.zeros(len(X[0]))
    epochs = 100
    train_accuracy = []
    test_accuracy = []

    for e in range(epochs):
        total_error = 0
        for i, x in enumerate(X_test):
            if(np.dot(X_test[i], w)*Y_test[i]) <= 0:
                total_error+=1
        test_accuracy.append(1 - total_error/5512)

        total_error = 0
        for i, x in enumerate(X_train):
            if(np.dot(X_train[i], w)*Y_train[i]) <= 0:
                total_error+=1
        train_accuracy.append(1 - total_error/8270)

        for i, x in enumerate(X_train):
            if(np.dot(X_train[i], w)*Y_train[i]) <= 0:
                w = w + X[i]*Y[i]

    total_error = 0
    for i, x in enumerate(X_test):
        if(np.dot(X_test[i], w)*Y_test[i]) <= 0:
            total_error+=1
    test_accuracy.append(1 - total_error/5512)

    total_error = 0
    for i, x in enumerate(X_train):
        if(np.dot(X_train[i], w)*Y_train[i]) <= 0:
            total_error+=1
    train_accuracy.append(1 - total_error/8270)


    plt.plot(train_accuracy)
    plt.plot(test_accuracy)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

    return w

print(perceptron(X_train, Y_train, X_test, Y_test))
