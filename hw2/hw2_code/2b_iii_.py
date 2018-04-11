import numpy as np
import random

X = np.loadtxt('train.csv', delimiter=',', skiprows=1)
X_test = np.loadtxt('test.csv', delimiter=',', skiprows=1)

def get_best_split(B, c):#feature_array, data_array

    N = len(B)

    best_split = -1
    max_reduction = float('-inf')

    for j in range(len(c)):#find the best split
        p = 0
        Nc = 0
        pc_l = 0
        pc_r = 0

        for i in range(N):
            if B[i][-1] == 1:
                p+=1
            if B[i][c[j]] <= 0.5:
                Nc+=1
                if B[i][-1] == 1:
                    pc_l+=1
            else:
                if B[i][-1] == 1:
                    pc_r+=1

        p/=N
        pc_l/=Nc
        pc_r/=(N-Nc)
        delta_I = 2*(p*(1-p) - Nc/N*(pc_l*(1-pc_l)) - (N-Nc)/N*(pc_r*(1-pc_r)))

        if delta_I > max_reduction:
            max_reduction = delta_I
            best_split = c[j]

        vote_l = 0. if pc_l<=0.5 else 1.
        vote_r = 0. if pc_r<=0.5 else 1.

    return (max_reduction, best_split, vote_l, vote_r)

def predict(best_split, X_test, vote_l, vote_r):
    prediction = []
    for i in range(len(X_test)):
        if X_test[i][best_split] <= 0.5:
            prediction.append(vote_l)
        else:
            prediction.append(vote_r)

    return prediction

def method1(X_test, prediction):
    error = 0
    for i in range(len(X_test)):
        #(more than half vote for 1 but real label is 0) or (more than half vote for 0 but real label is 1)
        if (prediction[i] > 500 and X_test[i][-1] == 0.) or (prediction[i] <= 500 and X_test[i][-1] == 1.):#more than half vote for 1 but real label is 0
            error+=1

    return error/len(X_test)

def method2(X_test, prediction):
    error = 0
    for i in range(len(X_test)):
        if prediction[i] != X_test[i][-1]:
            error+=1

    return error/len(X_test)

for k in range(1, 6):#k features
    total_prediction = [0]*len(X_test)
    error2 = 0
    for m in range(1000):#1000 stumps

        B = X[np.random.choice(X.shape[0], 400, replace=True), :]#randomly select 400 data with replacement
        c = random.sample(range(5), k)#randomly generate k features

        (max_reduction, best_split, vote_l, vote_r) = get_best_split(B, c)
        prediction = predict(best_split, X_test, vote_l, vote_r)

        total_prediction = [sum(x) for x in zip(total_prediction, prediction)]#sum up all the votes, preparing for method1
        error2+=method2(X_test, prediction)#sum up all the errors, preparing for method2

    print('k: ' + str(k))
    print('method1: ' + str(method1(X_test, total_prediction)))
    print('method2: ' + str(error2/1000))
    print()

