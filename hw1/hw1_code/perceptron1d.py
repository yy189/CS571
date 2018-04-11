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
    epochs = 100
    w_star = np.zeros(len(X[0]))

    for e in range(epochs):
        total_errors = 0
        for i, x in enumerate(X):
            if(np.dot(X[i], w_star)*Y[i]) <= 0:
                total_errors+=1
                w_star = w_star + X[i]*Y[i]
            if e == 0 and i == int(1/3 * X.shape[0]):
                w_prime = w_star
        if(total_errors == 0):
            return(w_prime, w_star)
    return (w_prime, w_star)

(w_prime, w_star) = perceptron(X_train, Y_train)

xy_arr_prime = []
xy_arr_star = []
intercepts = range(-1000, 1000, 10)

for j, b in enumerate(intercepts):
    TP_star = FP_star = FN_star = TN_star = TP_prime = FP_prime = FN_prime = TN_prime = 0
    for i, x in enumerate(X_train):
        if np.dot(X_train[i], w_prime)+intercepts[j] > 0:
            if(Y_train[i] == 1):
                TP_prime+=1
            else:
                FP_prime+=1
        else:
            if(Y_train[i] == 1):
                FN_prime+=1
            else:
                TN_prime+=1

        if np.dot(X_train[i], w_star)+intercepts[j] > 0:
            if(Y_train[i] == 1):
                TP_star+=1
            else:
                FP_star+=1
        else:
            if(Y_train[i] == 1):
                FN_star+=1
            else:
                TN_star+=1

    xy_arr_prime.append([FP_prime/(FP_prime+TN_prime), TP_prime/(TP_prime+FN_prime)])
    xy_arr_star.append([FP_star/(FP_star+TN_star), TP_star/(TP_star+FN_star)])


x_prime = [_v[0] for _v in xy_arr_prime]
y_prime  = [_v[1] for _v in xy_arr_prime]
x_star = [_v[0] for _v in xy_arr_star]
y_star = [_v[1] for _v in xy_arr_star]
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.plot(x_prime, y_prime, "X-", label="w'")
plt.plot(x_star, y_star, "+-", label="w*")
plt.show()

auc_prime = 0.
prev_x = 0
for x,y in xy_arr_prime:
    if x != prev_x:
        auc_prime += (x-prev_x)*y
        prev_x = x

auc_star = 0.
prev_x = 0
for x,y in xy_arr_star:
    if x != prev_x:
        auc_star += (x-prev_x)*y
        prev_x = x

print('auc_prime = ' + str(auc_prime) + ', auc_star = '  + str(auc_star))

