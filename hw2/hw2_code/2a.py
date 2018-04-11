import numpy as np

X = np.loadtxt('test.csv', delimiter=',', skiprows=1)

N=len(X)

def best_split():
    for c in range(len(X[0])-1):
        p = 0
        Nc = 0
        pc_l = 0
        pc_r = 0
        for i in range(N):
            if X[i][-1] == 1:
                p+=1
            if X[i][c] <= 0.5:
                Nc+=1
                if X[i][-1] == 1:
                    pc_l+=1
            else:
                if X[i][-1] == 1:
                    pc_r+=1

        print((pc_l + N-Nc-pc_r)/N)

        p/=N
        pc_l/=Nc
        pc_r/=(N-Nc)
        delta_I = 2*(p*(1-p) - Nc/N*(pc_l*(1-pc_l)) - (N-Nc)/N*(pc_r*(1-pc_r)))
        #print(delta_I)

def best_surrogate():
    for c in range(1, len(X[0])-1):
        pL = 0
        pR = 0
        PLL = 0
        PRR = 0

        for i in range(N):
            if X[i][0] <= 0.5:
                pL+=1
                if X[i][c] <= 0.5:
                    PLL+=1
            else:
                pR+=1
                if X[i][c] > 0.5:
                    PRR+=1

        pL/=N
        pR/=N
        PLL/=N
        PRR/=N
        _lambda = 1-(1-PLL-PRR)/min(pL, pR)
        print(_lambda)

best_split()
