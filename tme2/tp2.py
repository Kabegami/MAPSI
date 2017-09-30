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

def histogramme_binomiale(n):
    L = []
    for i in range(1000):
        v = binomiale(n, 0.5)
        L.append(v)
    table = np.array(L)
    res = plt.hist(table, n//2)
    plt.show()

def main():
    histogramme_binomiale(40)

main()
