import numpy as np
import random
import math

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

        vote_l = 0 if pc_l<=0.5 else 1
        vote_r = 0 if pc_r<=0.5 else 1

    return (max_reduction, best_split, vote_l, vote_r)

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



Q = [0.4, 0.5, 0.6, 0.7, 0.8]

for q in Q:

    variable_importance = [0, 0, 0, 0, 0]
    error_diff = [0, 0, 0, 0, 0]

    counts = [0, 0, 0, 0, 0]
    importance = []
    importance_OOB = []
    for i in range(5):
        importance.append([])
        importance_OOB.append([])

    for m in range(1000):#1000 stumps
        #bootstraping
        id = [i for i in range(len(X))]
        prob_rf=X.copy()
        prob_rf=np.insert(prob_rf, 0, id, axis=1)
        B = prob_rf[np.random.choice(prob_rf.shape[0], size=400, replace=True), :]#randomly select 400 data with replacement
        OOB = np.array(list(set(tuple(map(tuple, prob_rf)))-set(tuple(map(tuple, B)))))[:,1:]
        B = B[:,1:]

        c = random.sample(range(5), 2)#randomly generate 2 features

        (max_reduction, best_split, vote_l, vote_r) = get_best_split(B, c)
        variable_importance[best_split]+=max_reduction#update total reduction of the best split

        error_difference = raw_importance(best_split, OOB, vote_l, vote_r)
        error_diff[best_split]+=error_difference#update total error difference of the best split

        counts[best_split]+=1
        importance[best_split].append(max_reduction)
        importance_OOB[best_split].append(error_difference)

    #compute mean
    variable_importance = [x/y for x,y in zip(variable_importance, counts)]
    error_diff = [x/y for x,y in zip(error_diff, counts)]

    variance = [0, 0, 0, 0, 0]
    variance_OOB = [0, 0, 0, 0, 0]

    for i in range(5):
        for j in range(len(importance[i])):
            variance[i]+=(importance[i][j]-variable_importance[i])**2
            variance_OOB[i]+=(importance_OOB[i][j]-error_diff[i])**2

    #compute standard deviation
    standard_deviation = [math.sqrt(x/y) for x,y in zip(variance, counts)]
    standard_deviation_OOB = [math.sqrt(x/y) for x,y in zip(variance_OOB, counts)]


    print('q: ' + str(q))
    print('variable_importance: ' + str(variable_importance))
    print('variable_importance_OOB: ' + str(error_diff))

    print('standard deviation of variable_importance: ' + str(standard_deviation))
    print('standard deviation of variable_importance_OOB' + str(standard_deviation_OOB))

    print()

