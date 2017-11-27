# coding: utf-8

import random
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import math

np.set_printoptions(precision=2, linewidth=320,suppress=True)
plt.close('all')

def tracerLettre(let):
    a = -let*np.pi/180;
    coord = np.array([[0, 0]]);
    for i in range(len(a)):
        x = np.array([[1, 0]]);
        rot = np.array([[np.cos(a[i]), -np.sin(a[i])],[ np.sin(a[i]),np.cos(a[i])]])
        xr = x.dot(rot) # application de la rotation
        coord = np.vstack((coord,xr+coord[-1,:]))
    plt.plot(coord[:,0],coord[:,1])
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

def find_label(i, index):
    for label in range(len(index)):
        if i in index[label]:
            return label
    ValueError("Le label n'existe pas dans la base d'appprentissage")

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

def argMax(vecteur, A,j,debug=False):
    maxi = -1 * float('inf')
    index = None
    for i in range(len(vecteur)):
        transition = math.log(A[i][j])
        if debug:
            print('j : ', j)
            print('log transition ', transition)
            print('A[i] :', A[i])
            print('A[i][j]', A[i][j])
            print('vecteur :', vecteur)
        v = vecteur[i] + math.log(A[i][j])
        if debug:
            print('v : ', v)
            print('maxi :', maxi)
        if v >= maxi:
            maxi = v
            index = i
    if index == None:
        ValueError("Il n'y a pas d'index !")
    return index

def viterbi(X, Pi, A, B,debug=False):
    T = len(X)
    N, K = B.shape
    if debug:
        print('N, K', B.shape)
    # N : etats
    delta = np.zeros((N,T))
    psi = np.zeros((N,T))
    if debug:
        print('Pi taille :', Pi.shape)
    for i in range(N):
        delta[i][0] = math.log(Pi[i]) + math.log(B[i][X[0]])
        psi[i][0] = -1
    if debug:
        print('delta : ',delta)
    for t in range(1, T):
        #A chaque pas de temps
        for j in range(N):
            #Pour chaque etat
            pred = delta[:, t-1]
            if debug:
                print('pred : ', pred)
                print('A ', A)
            i_opt = argMax(pred, A, j,False)
            #on trouve son meilleur predecesseur
            if debug:
                print('i_opt :', i_opt)
            delta[j][t] = (delta[i_opt][t-1] + math.log(A[i_opt][j])) + math.log(B[j][X[t]])
            psi[j][t] = i_opt
    if debug:
        print('delta :', delta)

    S = np.max(delta[:,T-1])
    if debug:
        print('S : ', S)
    #backtring
    s = np.zeros(T)
    s[T-1] = np.argmax(delta[:,t])
    for t in range(T-2,1,-1):
        s[t] = psi[:,t+1][int(s[t+1])]
    if debug:
        print('s : ', s)
    return S, s

def printModel(M):
    print('==================== MODEL ===================')
    print(' pi : ',M[0])
    print('----------------------------------------------')
    print(' A : ', M[1])
    print('----------------------------------------------')
    print(' B : ' ,M[2])
    print('==============================================')
    

def Baum_Welch(X, Y, N, K):
    #initialisation
    S = initGD(X,N)
    index = groupByLabel(Y)
    converge = False
    cpt = 0
    old = float('inf')
    #print('S : ', S)
    #initialisation
    while not(converge):
        models = []
        L = 0
        for lettre in range(len(np.unique(Y))):
            M = learnHMM(X[index[lettre]], S[index[lettre]], N, K)
            models.append(M)
        #on parcours la base d'apprentissage
        for i in range(len(X)):
            xi = X[i]
            M =  models[find_label(i, index)]
            proba, s = viterbi(xi , M[0], M[1], M[2])
            S[i] = s.astype(int)
            L += proba
        if old == float('inf'):
            pass
        elif (old - L) / (1.0*old) < math.exp(-4):
            converge = True
        old = L
        print('iteration numero : ', cpt)
        print('vraisemblance :' ,old)
        cpt += 1
    return models
        
def predict(xi, models):
    maxi = -1* float('inf')
    index = None
    for lettre in range(len(models)):
        M = models[lettre]
        proba, s = viterbi(xi.astype(int), M[0], M[1], M[2])
        if proba > maxi:
            maxi = proba
            index = lettre
    return maxi, index

def accuracy(X, Y, models):
    K = len(X)
    Ynum = np.zeros(Y.shape)
    accuracy = 0
    vPred = np.zeros(K)
    for num,char in enumerate(np.unique(Y)):
        Ynum[Y == char] = num
    for i in range(K):
        xi = X[i]
        pred = predict (xi, models)[1]
        vPred[i] = pred
    a =  np.where(vPred != Ynum, 0.,1.).mean()
    print('accuracy : {} %'.format( a * 100))
    return a, vPred, Ynum

def draw(conf,Y):
    plt.figure()
    plt.imshow(conf, interpolation='nearest')
    plt.colorbar()
    plt.xticks(np.arange(26),np.unique(Y))
    plt.yticks(np.arange(26),np.unique(Y))
    plt.xlabel(u'Vérité terrain')
    plt.ylabel(u'Prédiction')
    plt.savefig("mat_conf_lettres.png")

def random_proba(P):
    """ la proba P est deja sous cumsum normalement """
    r = random.random()
    for i in range(len(P)):
        if r <= P[i]:
            return i
    return len(P) - 1
    
def generateHMM(Pi, A,B, datatype, longueur=10):
    sequence = []
    X = []
    so = random_proba(Pi)
    sequence.append(so)
    xo = random_proba(B[so])
    cpt =0
    previous = so
    while cpt < longueur:
        cpt += 1
        si = random_proba(A[previous])
        xi = random_proba(B[si])
        previous = si
        sequence.append(si)
        X.append(xi)
    return sequence, X
    

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
    #le resultat est légèrement différent !
    s_est, p_est = viterbi(Xd[0], Pi, A, B)
    print(s_est)
    print(p_est)
    #print('=============================================================')
    models = Baum_Welch(Xtrain, Ytrain, N, K)
    a,pred, Ynum = accuracy(Xtest,Ytest,models)
    conf = np.zeros((26,26))
    for i in range(len(Ynum)):
        conf[int(pred[i])][int(Ynum[i])] += 1
    draw(conf,Ynum)
    #================================================================
    #          PARTIE GENERATIVE (OPTIONNELLE)
    #================================================================
    n = 3          # nb d'échantillon par classe
    nClred = 5   # nb de classes à considérer
    fig = plt.figure()
    for cl in range(nClred):
        Pic = models[cl][0].cumsum() # calcul des sommes cumulées pour gagner du temps
        Ac = models[cl][1].cumsum(1)
        Bc = models[cl][2].cumsum(1)
        long = np.floor(np.array([len(x) for x in Xd[itrain[cl]]]).mean()) # longueur de seq. à générer = moyenne des observations
        for im in range(n):
            s,x = generateHMM(Pic, Ac, Bc, int(long))
            intervalle = 360./d  # pour passer des états => angles
            newa_continu = np.array([i*intervalle for i in x]) # conv int => double
            sfig = plt.subplot(nClred,n,im+n*cl+1)
            sfig.axes.get_xaxis().set_visible(False)
            sfig.axes.get_yaxis().set_visible(False)
            tracerLettre(newa_continu)
    plt.savefig("lettres_hmm.png")
    plt.show()

main()
