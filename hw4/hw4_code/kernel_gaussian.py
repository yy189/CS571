import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
import random
from sklearn import svm
from sklearn import preprocessing
from sklearn import metrics

def train(X_train, y_train, clf, _gamma):
    clf.set_params(kernel='rbf', C=1.0, gamma=_gamma).fit(X_train, y_train)

def predict(X_test, clf):
    y_score = clf.decision_function(X_test)
    prediction=clf.predict(X_test)
    return (prediction, y_score)

def accuracy(prediction, y_test):
    correct = 0
    for i, x in enumerate(prediction):
        if prediction[i] == y_test[i]:
        #if (y_score[i] > 0 and y_test[i] == 1) or (y_score[i] <= 0 and y_test[i] == 0):
            correct+=1
    print('accuracy = ' + str(correct/len(y_test)))

if __name__ == '__main__':

    X = np.loadtxt('creditCard.csv', delimiter=',', skiprows=1)
    random.Random(2018).shuffle(X)
    Y = X[:, -1]
    X = np.delete(X, -1, axis=1)
    #X = preprocessing.normalize(X, norm='l2')

    num = int(len(X)*0.9)
    X_train, X_test = X[:num], X[num:]
    y_train, y_test = Y[:num], Y[num:]


    plt.xlim(0,1)
    plt.ylim(0,1)

    clf = svm.SVC()
    train(X_train, y_train, clf, 1/5)
    prediction, y_score = predict(X_test, clf)
    print("Sigma square = 5: ")
    accuracy(prediction, y_test)
    fpr_5, tpr_5, thresholds_5 = metrics.roc_curve(y_test, y_score)
    plt.plot(fpr_5,tpr_5, 'r')
    print("auc:"+str(metrics.auc(fpr_5,tpr_5)))


    #xy_arr_5 = roc_auc(y_score, y_test)

    clf1 = svm.SVC()
    train(X_train, y_train, clf1, 1/25)
    prediction1, y_score1 = predict(X_test, clf1)
    print("Sigma square = 25: ")
    accuracy(prediction1, y_test)
    fpr_25, tpr_25, thresholds_25 = metrics.roc_curve(y_test, y_score1)
    plt.plot(fpr_25,tpr_25, 'b')
    print("auc:"+str(metrics.auc(fpr_25,tpr_25)))

    plt.show()


