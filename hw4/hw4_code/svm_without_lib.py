import numpy as np
import random
from sklearn import preprocessing
from cvxopt import matrix
import cvxopt.solvers as solvers

def test(x, w, b):
    return np.sign(np.dot(x, w)+b)

def train(x, y):
    n_samples, n_features = x.shape

    # Gram matrix
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i,j] = np.dot(x[i], x[j])

    P = matrix(np.outer(y,y) * np.inner(x,x))
    q = matrix(-np.ones((n_samples, 1)))
    G = matrix(np.eye(n_samples) * -1)
    h = matrix(np.zeros(n_samples))
    A = matrix(y.reshape(1, -1))
    b = matrix(np.zeros(1))
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)
    a = np.ravel(sol['x'])

    # Support vectors have non zero lagrange multipliers
    sv = a > 1e-10
    ind = np.arange(len(a))[sv]
    a = a[sv]
    sv_x = x[sv]
    sv_y = y[sv]

    # Weight vector
    w = np.zeros(n_features)
    for n in range(len(a)):
        w += a[n] * sv_y[n] * sv_x[n]

    cond = sv_y == 1
    b_ = sv_y[cond]-np.dot(sv_x[cond],w)
    if b_.size==0:
        return "false"
    b=b_[0]

    return (w, b)



if __name__ == '__main__':
    x_train = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    y_train = np.array([1., 0., 1.])

    print(train(x_train, y_train))


