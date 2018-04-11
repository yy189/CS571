import numpy as np
import random

X = np.loadtxt('train.csv', delimiter=',', skiprows=1)

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


def get_best_surrogate(B, c, best_split):

    N = len(B)

    best_surrogate = -1
    max_lambda = float('-inf')

    for j in range(len(c)):

        if c[j] == best_split:#best split can't be best surrogate split
            continue

        pL = 0
        pR = 0
        PLL = 0
        PRR = 0

        for i in range(N):
            if X[i][0] <= 0.5:
                pL+=1
                if X[i][c[j]] <= 0.5:
                    PLL+=1
            else:
                pR+=1
                if X[i][c[j]] > 0.5:
                    PRR+=1

        pL/=N
        pR/=N
        PLL/=N
        PRR/=N
        _lambda = 1-(1-PLL-PRR)/min(pL, pR)

        if _lambda > max_lambda:
            max_lambda = _lambda
            best_surrogate = c[j]

    return best_surrogate

def raw_importance(best_split, OOB, vote_l, vote_r):

    OOB_permuted = OOB.copy()
    OOB_permuted = np.transpose(OOB_permuted)#randomly permute only the (best_split)th feature
    OOB_permuted[best_split] = np.random.permutation(OOB_permuted[best_split])
    OOB_permuted = np.transpose(OOB_permuted)

    error = 0
    error_permuted = 0

    for i in range(len(OOB)):
        #count error for OOB
        if OOB[i][best_split] <= 0.5:
            if vote_l != OOB[i][-1]:
                error+=1
        else:
            if vote_r != OOB[i][-1]:
                error+=1

        #count error for OOB_permuted
        if OOB_permuted[i][best_split] <= 0.5:
            if vote_l != OOB[i][-1]:
                error_permuted+=1
        else:
            if vote_r != OOB[i][-1]:
                error_permuted+=1

    return (error_permuted - error)/len(OOB)



for k in range(1, 6):#k features

    feature_as_best_split = [0, 0, 0, 0, 0]
    feature_as_best_surrogate = [0, 0, 0, 0, 0]

    variable_importance = [0, 0, 0, 0, 0]
    error_diff = [0, 0, 0, 0, 0]

    counts = [0, 0, 0, 0, 0]

    for m in range(1000):#1000 stumps
        #bootstraping
        id = [i for i in range(len(X))]
        prob_rf=X.copy()
        prob_rf=np.insert(prob_rf, 0, id, axis=1)
        B = prob_rf[np.random.choice(prob_rf.shape[0], size=400, replace=True), :]#randomly select 400 data with replacement
        OOB = np.array(list(set(tuple(map(tuple, prob_rf)))-set(tuple(map(tuple, B)))))[:,1:]
        B = B[:,1:]

        c = random.sample(range(5), k)#randomly generate k features

        (max_reduction, best_split, vote_l, vote_r) = get_best_split(B, c)
        variable_importance[best_split]+=max_reduction#update total reduction of the best split

        best_surrogate = get_best_surrogate(B, c, best_split)

        feature_as_best_split[best_split]+=1
        if best_surrogate != -1:#when k==1, there is no best surrogate split
            feature_as_best_surrogate[best_surrogate]+=1

        error_diff[best_split]+=raw_importance(best_split, OOB, vote_l, vote_r)#update total error difference of the best split

        counts[best_split]+=1#update the number of times for being the best split

    for i in range(5):
        if counts[i] != 0:
            variable_importance[i]/=counts[i]
            error_diff[i]/=counts[i]
        else:
            variable_importance[i] = error_diff[i] = 'NA'


    print('k: ' + str(k))
    print('feature_as_best_split: ' + str(feature_as_best_split))
    print('feature_as_best_surrogate: ' + str(feature_as_best_surrogate))

    print('variable_importance: ' + str(variable_importance))
    print('variable_importance_OOB: ' + str(error_diff))

    print()

