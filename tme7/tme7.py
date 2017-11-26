# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import math

np.set_printoptions(precision=2, linewidth=320,suppress=True)
plt.close('all')

def tracerLettre(let, name="exlettre.png"):
    a = -let*np.pi/180; # conversion en rad
    coord = np.array([[0, 0]]); # point initial
    for i in range(len(a)):
        x = np.array([[1, 0]]);
        rot = np.array([[np.cos(a[i]), -np.sin(a[i])],[ np.sin(a[i]),np.cos(a[i])]])
        xr = x.dot(rot) # application de la rotation
        coord = np.vstack((coord,xr+coord[-1,:]))
    plt.figure()
    plt.plot(coord[:,0],coord[:,1])
    plt.savefig("images/" + name)
    plt.show()
    return

def discretise(X, d):
    intervalle = 360 / d
    M = []
    for xi in X:
        line = (np.floor(xi / intervalle)).astype(int)
        M.append(line)
    return np.array(M)

def groupByLabel( y):
    index = []
    for i in np.unique(y): # pour toutes les classes
        ind, = np.where(y==i)
        index.append(ind)
    return index

def separeTrainTest(y, pc):
    indTrain = []
    indTest = []
    for i in np.unique(y): # pour toutes les classes
        ind, = np.where(y==i)
        n = len(ind)
        indTrain.append(ind[np.random.permutation(n)][:(int)(np.floor(pc*n))])
        indTest.append(np.setdiff1d(ind, indTrain[-1]))
    return indTrain, indTest

def initGD(X, N):
    S = []
    for xi in X:
        v = np.floor(np.linspace(0,N-.00000001,len(xi)))
        S.append(v.astype(int))
    return np.array(S)

def test_initGD():
    X = np.array([ 1,  9,  8,  8,  8,  8,  8,  9,  3,  4,  5,  6,  6,  6,  7,  7,  8,  9,  0,  0,  0,  1,  1])
    S = [ 0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3]
    S2 = initGD([X],4)
    #print(S2[0])
    np.testing.assert_almost_equal(S, S2[0])

def learnHMM(allX, allS, N, K, initTo0=False):
    if initTo0:
        A = np.zeros((N,N))
        B = np.zeros((N,K))
        Pi = np.zeros(N)
    else:
        eps = 1e-8
        A = np.ones((N,N))*eps
        B = np.ones((N,K))*eps
        Pi = np.ones(N)*eps
    for i in range(len(allX)):
        xi =allX[i]
        si =allS[i]
        Pi[si[0]] += 1
        for i in range(0,len(si)-1):
            etat = si[i]
            next_etat= si[i+1]
            A[etat][next_etat] += 1
            observation = xi[i]
            B[etat][observation] += 1
            
    A = A/np.maximum(A.sum(1).reshape(N,1),1) # normalisation
    Pi = Pi/Pi.sum()
    M = []
    for line in B:
        nb = np.sum(line)
        line = line / (1.0*nb)
        M.append(line)
    
    return Pi, A, np.array(M)

def viterbi(X, Pi, A, B):
    d0 = log(Pi) + log(bi[x[0]])
    fi0 = -1
    

        
        
        
    
def main():
    with open('lettres.pkl', 'rb') as f:
        data = pkl.load(f, encoding='latin1')
    X = np.array(data.get('letters'))
    Y = np.array(data.get('labels'))
    d = 21
    nCl = 26
    K = 10
    N = 5
    itrain,itest = separeTrainTest(Y,0.8)
    ia = []
    for i in itrain:
        ia += i.tolist()    
    it = []
    for i in itest:
        it += i.tolist()
        
    Xd = discretise(X,K)
    
    Xtrain = Xd[ia]
    Xtest = Xd[it]

    Ytrain = Y[ia]
    Ytest = Y[it]
    test_initGD()
    S = initGD(X, N)
    b = Xd[Y=='a']
    #print('b : ', b)
    Pi, A, B = learnHMM(Xd[Y=='a'], S[Y=='a'],N,K)
    print('Pi : ', Pi)
    print('A :', A)
    print('B : ', B)

main()
