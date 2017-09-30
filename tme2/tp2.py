import random
import math
import numpy as np
import matplotlib.pyplot as plt
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
    print(4 * sigma)
    print(1 * k)
    q = (4 * sigma) / (k * 1.0)
    x = -2 * sigma
    print("q : ", q)
    while x < 2*sigma:
        y = N(x, sigma, u)
        L_yi.append(y)
        L_xi.append(x)
        x += q
        #print(x)
    y = N(x, sigma, u)
    L_yi.append(y)
    L_xi.append(x)
    return L_yi, L_xi

def plot_normale(k, sigma,u=0):
    yi, xi = normale(k,sigma,u)
    #print(L)
    plt.plot(xi, yi)
    plt.show()

def histogramme_binomiale(n):
    L = []
    for i in range(1000):
        v = binomiale(n, 0.5)
        L.append(v)
    table = np.array(L)
    res = plt.hist(table, n)
    plt.show()

def main():
    #histogramme_binomiale(20)
    plot_normale(1001,1)

main()
