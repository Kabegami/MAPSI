# coding: utf-8

import random
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.close('all')

def combinaison(k,n):
    return math.factorial(n) / (math.factorial(k) * math.factorial(n-k)*1.0)

def bernoulli(p):
    v = random.random()
    if v <= p:
        return 1
    return 0

def binomiale(n, p):
    L = []
    for k in range(n+1):
        # on ajoute P(X = k) dans L
        proba = combinaison(k,n)*(p**k)*((1-p)**(n-k))
        L.append(proba)
    r = random.random()
    t = 0
    for i in range(len(L)):
        proba = L[i]
        t += proba
        if r < t:
            return i
    return n

def N(x, sigma, u=0):
    l =  1 / (math.sqrt(2*math.pi) * sigma)
    r = math.exp((-1/2)*(((x-u)/sigma)**2))
    return l * r

def normale(k, sigma, u=0):
    if k % 2 == 0:
        raise ValueError ('Le nombre k doit etre impair')
    L_yi = []
    L_xi = []
    q = (4 * sigma) / (k * 1.0)
    x = -2 * sigma
    while x < 2*sigma:
        y = N(x, sigma, u)
        L_yi.append(y)
        L_xi.append(x)
        x += q
    y = N(x, sigma, u)
    L_yi.append(y)
    L_xi.append(x)
    return L_yi, L_xi

def plot_normale(k, sigma,u=0):
    yi, xi = normale(k,sigma,u)
    #print(L)
    plt.plot(xi, yi)
    plt.show()

def proba_affine(k, slope, epsilon=0.00001):
    if k % 2 == 0:
        raise ValueError('Le nombre k doit etre impair')
    if abs(slope) > 2.0 /( k*k):
        raise ValueError('La pente est trop raide : pente max = {}'.format(2.0 / (k * k)))
    y = []
    x = []
    s = 0
    for xi in range(k):
        yi = (1.0 / k) + (xi - ((k - 1)/2)) * slope
        s += yi
        y.append(yi)
        x.append(xi)
    if (s > 1 + epsilon) or (s < 1 - epsilon):
        raise ValueError('La probabilité ne somme pas à 1 : s = {}'.format(s))
    return y, x

def plot_affine(k, pente):
    y,x = proba_affine(k,pente)
    plt.plot(x,y)
    plt.show()
    
    

def histogramme_binomiale(n,p=0.5):
    L = []
    for i in range(1000):
        v = binomiale(n, p)
        L.append(v)
    table = np.array(L)
    res = plt.hist(table, n)
    plt.show()

def Pxy(x, y):
    """ float np.array x float np.array -> float np.2D-array """
    L = []
    for xi in x:
        line = []
        for yi in y:
            line.append(xi * yi)
        L.append(line)
    return np.array(L)

def dessine(P_jointe) :
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.linspace(-3, 3, P_jointe.shape[0])
    y = np.linspace(-3, 3, P_jointe.shape[1])
    print('x : {}, y : {}'.format(x,y))
    X, Y = np.meshgrid(x, y)
    print('X : {}, Y : {}'.format(X,Y))
    ax.plot_surface(X, Y, P_jointe, rstride=1, cstride=1 )
    ax.set_xlabel('A')
    ax.set_ylabel('B')
    ax.set_zlabel('P(A) * P(B) ')
    plt.show()
            
    
def main():
    #histogramme_binomiale(20)
    #plot_normale(1001,1)
    #plot_affine(21,0.001)
    PA = np.array(normale(21,1)[0])
    #print(PA)
    PB = np.array(proba_affine(21,0.00001)[0])
    #print(PB)
    Pjointe = Pxy(PA, PB)
    #print(Pjointe)
    dessine(Pjointe)
    print(res)

main()
