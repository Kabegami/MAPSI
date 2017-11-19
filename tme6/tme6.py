# coding: utf-8

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

def learnMarkovModel(Xc, d):
    A = np.zeros((d,d))
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
    L = math.log(Pi[s[0]])
    for i in range(1, len(s)):
        si = s[i]
        previous = s[i-1]
        L += A[previous][si]
    return L

def main():
    d=3
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
    plt.show()

main()
