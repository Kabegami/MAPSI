# coding: utf-8

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle as pkl
import numpy.random as npr
import math

def memo(f):
    dico = dict()
    def helper(mess, *args):
        if mess not in dico:
            dico[mess] = f(mess, *args)
        return dico[mess]
    return helper
        

def tirage(m):
    """float -> float x float"""
    r1 = random.uniform(-m,m)
    r2 = random.uniform(-m,m)
    return r1, r2

def monteCarlo(N):
    """int -> float x float np.array x float np.array"""
    print('N : ', N)
    pi = 0
    cptR = 0
    cptC = 0
    X = np.zeros(N)
    Y = np.zeros(N)
    for i in range(0, N):
        x, y = tirage(1)
        X[i] = x
        Y[i] = y
        if x**2 + y ** 2 < 1:
            cptR += 1
        else:
            cptC += 1
    pi = (4 * cptR) / 1.0 * cptC
    return pi, X, Y

def affiche():
    plt.figure()

    # trace le carré
    plt.plot([-1, -1, 1, 1], [-1, 1, 1, -1], '-')

    # trace le cercle
    x = np.linspace(-1, 1, 100)
    y = np.sqrt(1- x*x)
    plt.plot(x, y, 'b')
    plt.plot(x, -y, 'b')

    # estimation par Monte Carlo
    pi, x, y = monteCarlo(int(1e4))
    print('pi : ', pi)
    print('x ' ,x)
    print('y : ', y)

    # trace les points dans le cercle et hors du cercle
    dist = x*x + y*y 
    plt.plot(x[dist <=1], y[dist <=1], "go")
    plt.plot(x[dist>1], y[dist>1], "ro")
    plt.show()

def swapF(d):
    """ (char,char) dict -> (char,char) dict """
    keys = d.keys()
    K = len(keys)
    c1 = random.randint(0, K-1)
    c2 = random.randint(0, K-1)
    new = d.copy()
    k1 = keys[c1]
    k2 = keys[c2]
    temp = d[k1]
    new[k1] = new[k2]
    new[k2] = temp
    return new

def decrypt(mess, d):
    """string x (char,char) dict -> string """
    s = ""
    for c in mess:
        s += d[c]
    return s

@memo
def logLikelihood(mess, mu, A, chars):
    """ string x float np.array x float np.2D-array x char list -> string 
    Notre model est une chiane de caractère de parametre Pi0 = mu et A"""
    i = chars.index(mess[0])
    L = math.log(mu[i])
    prec = mess[0]
    for i in range(1, len(mess)):
        k = chars.index(mess[i])
        k_prec = chars.index(prec)
        L += math.log(A[k_prec][k])
        prec = mess[i]
    return L
        
def MetropolisHastings(mess, mu, A, tau, N, chars):
    """ mess : message codé
        mu, A : parametre de la chaine de Markov
        tau : fonction de décodage
        N : nombre max d'itérations """
    best_mess = mess
    best_liklihood = logLikelihood(mess, mu, A, chars)
    old =  best_liklihood
    for i in range(N):
        if i % 1000 == 0:
            print('i : ', i)
        new = swapF(tau)
        #print('mess :', type(mess))
        new_mess = decrypt(mess, new)
        L = logLikelihood(new_mess,mu,A,chars)
        alpha = min(1, L / old)
        #print('L : ', L)
        #print('best : ', best_liklihood)
        old = L
        #print('alpha :' , alpha)
        r = random.random()
        if r <= alpha:
            #transition accepté
            tau = new.copy()
            if L > best_liklihood:
                print('boucle')
                best_mess = decrypt(mess, tau)
                best_liklihood = L
    return best_mess
                
        

def identityTau (count):
    tau = {} 
    for k in count.keys ():
        tau[k] = k
    return tau

def updateOccurrences(text, count):
   for c in text:
      if c == u'\n':
         continue
      try:
         count[c] += 1
      except KeyError as e:
         count[c] = 1

def mostFrequent(count):
   bestK = []
   bestN = -1
   for k in count.keys():
      if (count[k]>bestN):
         bestK = [k]
         bestN = count[k]
      elif (count[k]==bestN):
         bestK.append(k)
   return bestK

def replaceF(f, kM, k):
   try:
      for c in f.keys():
         if f[c] == k:
            f[c] = f[kM]
            f[kM] = k
            return
   except KeyError as e:
      f[kM] = k

def mostFrequentF(message, count1, f={}):
   count = dict(count1)
   countM = {}
   updateOccurrences(message, countM)
   while len(countM) > 0:
      bestKM = mostFrequent(countM)
      bestK = mostFrequent(count)
      if len(bestKM)==1:
         kM = bestKM[0]
      else:
         kM = bestKM[npr.random_integers(0, len(bestKM)-1)]
      if len(bestK)==1:
         k = bestK[0]
      else:
         k = bestK[npr.random_integers(0, len(bestK)-1)]
      replaceF(f, kM, k) 
      countM.pop(kM)
      count.pop(k)
   return f

def save(message,fname):
    f = open(fname, 'w')
    f.write(message)
    f.close()
    

def main():
    #affiche()
    (count, mu, A) = pkl.load(file("countWar.pkl", "rb"))
    secret = (open("secret.txt", "r")).read()[0:-1] # -1 pour supprimer le saut de ligne
    secret2 = (open("secret2.txt", "r")).read()[0:-1] # -1 pour supprimer le saut de ligne
    tau = {'a' : 'b', 'b' : 'c', 'c' : 'a', 'd' : 'd' }
    print('tau : ', tau)
    swapF(tau)
    print('new tau : ', tau)
    s1 = decrypt ('aabcd', tau)
    print('s1 : ', s1)
    s2 = decrypt ('dcba', tau)
    print('s2 : ', s2)
    v1 = logLikelihood('abcd', mu, A, count.keys())
    print('v1 : ', v1)
    v2 = logLikelihood( "dcba", mu, A, count.keys () )
    print('v2 :', v2)
    print('typescecret : ', type(secret2))
    #m = MetropolisHastings( secret2, mu, A, identityTau (count), 10000, count.keys())
    tau_init = mostFrequentF(secret2, count, identityTau (count) )
    m = MetropolisHastings(secret2, mu, A, tau_init, 100000, count.keys() )
    print(m)
    save(m, 'decodage.txt')

main()
