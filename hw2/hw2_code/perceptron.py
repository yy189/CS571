import numpy as np

X = np.array([
    [.75, .10],
    [.85, .80],
    [.85, .95],
    [.15, .10],
    [.05, .25],
    [.05, .50],
    [.85, .25],
])

y = np.array([-1, -1, 1, -1, 1, 1, -1])

def perceptron(X, Y):
    w = np.zeros(len(X[0]))
    epochs = 1

    for e in range(epochs):
        for i, x in enumerate(X):
            if np.dot(X[i], w)*Y[i] <= 0:
                w = w + X[i]*Y[i]

    return w

w = perceptron(X, y)

error = 0
for i, x in enumerate(X):
    if np.dot(X[i], w) * y[i] <= 0:
        error += 1

print('error: ' + str(error) + ', w: ' + str(w))

