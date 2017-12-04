# coding: utf-8

import random
import math
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

with open('TME6_lettres.pkl', 'rb') as f:
    data = pkl.load(f, encoding='latin1') 
X = np.array(data.get('letters')) # récupération des données sur les lettres
Y = np.array(data.get('labels')) # récupération des étiquettes associées

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
        line = np.floor(xi / intervalle)
        M.append(line)
    return np.array(M)

def groupByLabel( y):
    index = []
    for i in np.unique(y): # pour toutes les classes
        ind, = np.where(y==i)
        index.append(ind)
    return index

def learnMarkovModel(Xc, d,init=False):
    if init:
        A = np.ones((d,d))
    else:
        A = np.zeros((d,d))/10**(-3)
    Pi = np.zeros(d)
    for x in Xc:
        #on parcours la base d'apprentissage
        Pi[(int)(x[0])] += 1
        for i in range(len(x)-1):
            #on parcours la representation de la lettre
            etat = (int)(x[i])
            next_etat = (int)(x[i+1])
            A[etat][next_etat] += 1

    #normalisation
    A = A/np.maximum(A.sum(1).reshape(d,1),1) # normalisation
    Pi = Pi/Pi.sum()
    return Pi, A

def probaSequence(s, Pi, A):
    """ remarque : les probabilitées à -inf viennent du fait que log(0) n'est pas définit dans la fonction log et qu'il y a une asymptote vers 0 """
    #print(' s :' , s)
    i0 = (int)(s[0])
    if Pi[i0] == 0:
        return -1 * float('inf')
    L = math.log(Pi[i0])
    for i in range(1, len(s)):
        si = (int)(s[i])
        previous = (int)(s[i-1])
        a= A[previous][si]
        if a == 0:
            return -1 * float('inf')
        L += math.log(a)
    return L

# separation app/test, pc=ratio de points en apprentissage
def separeTrainTest(y, pc):
    indTrain = []
    indTest = []
    for i in np.unique(y): # pour toutes les classes
        ind, = np.where(y==i)
        n = len(ind)
        indTrain.append(ind[np.random.permutation(n)][:(int)(np.floor(pc*n))])
        indTest.append(np.setdiff1d(ind, indTrain[-1]))
    return indTrain, indTest

def verif_log_proba(Xd, models):
    P = []
    for i in range(len(models)):
        Pi, A = models[i]
        value = probaSequence(Xd[0], Pi, A)
        P.append(value)
    P = np.array(P)
    print('P : ', P)

def graphe_accuracy_d(d_init, step, n,v=False):
    X = []
    Y = []
    xi = d_init
    while xi < n + step:
        X.append(xi)
        Y.append(separate_data_version(xi,v)[0])
        xi += step
    plt.plot(X,Y)
    plt.xlabel("discretisation d")
    plt.ylabel("accuracy")
    plt.show()

def separate_data_version(d, v=False):
    """ accuracy en initialisant la matrice A avec des 0 : 31 % 
        accuracy en initialisant la matrice A avec des 1 : 70 %"""
    itrain,itest = separeTrainTest(Y,0.8)
    #fusion des indices
    ia = []
    for i in itrain:
        ia += i.tolist()    
    it = []
    for i in itest:
        it += i.tolist()
#    print('ia : ', ia)
    Xtrain = []
    Xtest = []
    Xd = discretise(X, d)
    for i in range(len(Xd)):
        if i in ia:
            Xtrain.append(Xd[i])
        else:
            Xtest.append(Xd[i])
    Xtrain = np.array(Xtrain)
    Xtest =np.array(Xtest)
    
    Ytrain = Y[ia]
    Ytest = Y[it]
    index = groupByLabel(Ytrain)
    #print('Ytrain : ', Ytrain)
    #print('Xtrain : ', Xtrain)
    models = []
    Pi_sum = np.zeros(d)
    for cl in range(len(np.unique(Y))):
        #probleme, index[cl] contient aussi les index appartenant à Xtest
        models.append(learnMarkovModel(Xtrain[index[cl]], d,init=True))

    Pi_sum = np.zeros(d)
    for (Pi, A) in models:
        Pi_sum += Pi
    Pi_sum = Pi_sum / (1.0*len(Xtrain))
    #max de vraisemblance
    proba = np.array([[probaSequence(Xtest[i], models[cl][0], models[cl][1]) for i in range(len(Xtest))]for  cl in range(len(np.unique(Ytest)))])
    #print('proba : ', proba)
    #evaluation
    Ynum = np.zeros(Ytest.shape)
    for num,char in enumerate(np.unique(Ytest)):
        #il faut faire en sorte de ne pas avoir de problème d'index (mettre Ytest bug)
        Ynum[Ytest==char] = num

    pred = proba.argmax(0) # max colonne par colonne

    accuracy =  np.where(pred != Ynum, 0.,1.).mean()
    
    if v:
        print ('acuracy : ', accuracy)
    return accuracy, proba, Ynum, Pi_sum

def draw():
    conf = np.zeros((26,26))
    accuracy, proba, Ynum, Pi_sum = separate_data_version(20)
    pred = proba.argmax(0)
    for i in range(len(Ynum)):
        print('pred i', pred[i])
        print('Ynum i', Ynum[i])
        conf[int(pred[i])][int(Ynum[i])] += 1
    plt.figure()
    plt.imshow(conf, interpolation='nearest')
    plt.colorbar()
    plt.xticks(np.arange(26),np.unique(Y))
    plt.yticks(np.arange(26),np.unique(Y))
    plt.xlabel(u'Vérité terrain')
    plt.ylabel(u'Prédiction')
    plt.savefig("graphes/mat_conf_lettres.png")

def random_proba(Pi):
    sc = np.cumsum(Pi)
    r = random.random()
    for i in range(len(sc)):
        if r <= sc[i]:
            return i
    return len(sc) - 1
    
def generate(Pi, A, longeur):
    sequence = []
    so = random_proba(Pi)
    sequence.append(so)
    cpt = 0
    previous = so
    while cpt < longeur:
        cpt +=1
        si = random_proba(A[previous])
        previous = si
        sequence.append(si)
    print('sequence :', sequence)
    return sequence

def genere_alphabet(models,k,d):
    intervalle = 360./d # pour passer des états => valeur d'angles
    for m in models:
        s = generate(m[0], m[1],k)
        s_continu =  np.array([i*intervalle for i in s])
        tracerLettre(s_continu)
        
    
    
def main():
    d=20
    Xd = discretise(X, d)
    print("Xd 0 : ", Xd[0])
    #print("Y :", Y)
#    tracerLettre(X[0],'toto')
    index = groupByLabel(Y)
    #print('index :', index)
    models = []
    for cl in range(len(np.unique(Y))):
        #print('signaux cl :' , Xd[index[cl]])
        models.append(learnMarkovModel(Xd[index[cl]], d))
    print('models 0 ', models[0])

    verif_log_proba(Xd, models)

    
    proba = np.array([[probaSequence(Xd[i], models[cl][0], models[cl][1]) for i in range(len(Xd))]for  cl in range(len(np.unique(Y)))])
    print('proba : ', proba)
    Ynum = np.zeros(Y.shape)
    for num,char in enumerate(np.unique(Y)):
        Ynum[Y==char] = num

    pred = proba.argmax(0) # max colonne par colonne
    
    print ('acuracy : ', np.where(pred != Ynum, 0.,1.).mean())
    separate_data_version(d)
    graphe_accuracy_d(1,1,20,True)
    draw()
    #genere_alphabet(models, 25,d)
    newa = generate(models[0][0],models[0][1], 25) # generation d'une séquence d'états
    intervalle = 360./d # pour passer des états => valeur d'angles
    newa_continu = np.array([i*intervalle for i in newa]) # conv int => double
    print('lettre :', newa_continu)
    tracerLettre(newa_continu)
    plt.show()

main()
