import numpy as np
from sklearn.datasets import fetch_mldata
import matplotlib as mpl
mpl.use('TkAgg')
import math
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

def balanced_winnow(X, Y):
    w_p = np.full(len(X[0]), 1/(2*len(X[0])))
    w_n = np.full(len(X[0]), 1/(2*len(X[0])))
    eta=0.1
    epochs = 100
    accuracy = []

    for e in range(epochs):
        total_error = 0
        for i, x in enumerate(X):
            if (np.dot(X[i], w_p)-np.dot(X[i], w_n))*Y[i] <= 0:
                total_error+=1
        accuracy.append(1 - total_error/8270)

        for i, x in enumerate(X):
            if (np.dot(X[i], w_p)-np.dot(X[i], w_n))*Y[i] <= 0:
                w_p = w_p*math.e**(eta*Y[i]*X[i])
                w_n = w_n*math.e**(-eta*Y[i]*X[i])
                s = 0
                for j, w in enumerate(w_p):
                    s = s+w_p[j]+w_n[j]
                w_p /= s
                w_n /= s

    total_error = 0
    for i, x in enumerate(X):
        if (np.dot(X[i], w_p)-np.dot(X[i], w_n))*Y[i] <= 0:
            total_error+=1
    accuracy.append(1 - total_error/8270)

    delta = float('inf')
    for i, x in enumerate(X):
        temp = (np.dot(X[i], w_p)-np.dot(X[i], w_n))*Y[i]
        if temp > 0 and temp < delta:
            delta = temp

    new_eta = 1/2*np.log((1+delta)/(1-delta))
    print(new_eta)

    print(accuracy)

    plt.plot(accuracy)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

    return (w_p, w_n)

(w_p, w_n) = balanced_winnow(X_train, Y_train)

TP = FP = FN = TN = 0

for i, x in enumerate(X_test):
    if np.dot(X_test[i], w_p)-np.dot(X_test[i], w_n) > 0:
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
