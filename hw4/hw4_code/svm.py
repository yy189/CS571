import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
import random
from sklearn import svm
from sklearn import preprocessing
from sklearn import metrics

def train(X_train, y_train, clf):
    clf.set_params(kernel='linear', C=1.0).fit(X_train, y_train)

def predict(X_test, clf):
    y_score = clf.decision_function(X_test)
    prediction=clf.predict(X_test)
    return (prediction, y_score)

if __name__ == '__main__':
    X = np.loadtxt('creditCard.csv', delimiter=',', skiprows=1)
    random.Random(2018).shuffle(X)
    Y = X[:, -1]
    X = np.delete(X, -1, axis=1)
    #X = preprocessing.normalize(X, norm='l2')

    num = int(len(X)*0.9)
    X_train, X_test = X[:num], X[num:]
    y_train, y_test = Y[:num], Y[num:]

    clf = svm.SVC()
    train(X_train, y_train, clf)
    prediction, y_score = predict(X_test, clf)

    correct = 0
    for i, x in enumerate(y_score):
        if prediction[i] == y_test[i]:
        #if (y_score[i] > 0 and y_test[i] == 1) or (y_score[i] <= 0 and y_test[i] == 0):

            correct+=1
    print('accuracy = ' + str(correct/len(y_test)))

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)
    plt.plot(fpr,tpr)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.show()
    print("auc:"+str(metrics.auc(fpr,tpr)))
