#coding: utf8 

import numpy as np
import matplotlib.pyplot as plt
import random
import math

#============================================================
#                 FONCTIONS
#============================================================

def toy_data(a,b,N,sig):
    X = []
    Y = []
    epsilon = (sig) * np.random.randn(1,N)
    #print('epsilon : ', epsilon)
    for i in range(N):
        xi = random.random()
        yi = a * xi + b + epsilon[0][i]
        X.append(xi)
        Y.append(yi)
    return np.array(X), np.array(Y)

def toy_data2(a,b,c,N,sig):
    X = []
    Y = []
    epsilon = (sig) * np.random.randn(1,N)
    for i in range(N):
        xi = random.random()
        yi = a * (xi ** 2) + b * xi + c + epsilon[0][i]
        X.append(xi)
        Y.append(yi)
    return np.array(X), np.array(Y)

def draw(X,Y):
    plt.plot(X,Y, 'ro')
    plt.show()

def Mdraw(X,Y,model, l=None):
    plt.plot(X,Y, 'ro')
    if l is not None:
        plt.plot(X,model, label=l)
        plt.legend()
    else:
        plt.plot(X,model)
    #plt.xlabel('X')
    #plt.ylabel('Y')
    plt.show()

def E(X):
    return (sum(X) / float(len(X)))
        

def Cov(X,Y,Ex, Ey):
    """ Covariance
 Cov(X,Y) = E[(X - E(X))*(Y-E(Y))] """
    if len(X) != len(Y):
        raise ValueError("Les 2 variables aléatoires doivent avoir le meme nombre de valeur")
    Z = []
    for i in range(len(X)):
        Z.append((X[i] - Ex) * (Y[i] - Ey))
    return E(Z)
        

def estimate_parameter(X,Y):
    Ex = E(X)
    Ey = E(Y)
    sigmaX = 0
    for xi in X:
        sigmaX += (xi - Ex)**2
    sigmaX = sigmaX / len(X)
    print('Ex : ',Ex)
    print('Ey : ', Ey)
    cov = Cov(X,Y,Ex, Ey)
    print('cov : ', cov)
    a = cov / (float(sigmaX))
    b = Ey - a * Ex
    return a, b

def moindre_carre(Lx,Ly,biais=True):
    N = len(Lx)
    x = np.array(Lx)
    y = np.array(Ly)
    if biais:
        X = np.hstack((x.reshape(N,1),np.ones((N,1))))
        Y = np.hstack((y.reshape(N,1),np.ones((N,1))))
    else:
        X = x
        Y = y
    A = np.dot(X.T, X)
    B = np.dot(X.T, Y)
    #np.linag.solve attend des array de dimention 2
    w = np.linalg.solve(A,B)
    print('w :', w)
    a = w[0][0]
    b = w[1][0]
    return a,b

def cout_carree(x, y, w):
    y1 = np.dot(x,w)
    print('y1 : ', y1)
    e = y1 - y
    return np.dot(e.T, e)


def descente_gradient(x,y, n=30, cout=cout_carree, epsilon=5*10**(-3), h=10**(-3)):
    """ Soit C la fonction cout par defaut on a C = somme(ei) avec ei la diff au carre """
    w = np.zeros(x.shape[1])
    print('w : ', w)
    e0 = cout(x,y,w)
    #print('e0 : ', e0)
    allw = [w]
    #print('w : ', w)
    #print('vh :' , vh)
    for i in range(n):
        #calcul de dérivée numérique
        d = []
        for i in range(0, x.shape[1]):
            w2 = w.copy()
            w2[i] += h
            v = (cout(x,y,w2) - cout(x,y,w)) / h
            d.append(v)
        d = np.array(d)
        print('d : ', d)
        w = w - epsilon * d
        allw.append(w)
        #print('w : ',w)
    print('w : ', w)
    allw = np.array(allw)
    return w[0], w[1], allw
        

def gradient_draw():
    # tracer de l'espace des couts
    ngrid = 20
    w1range = np.linspace(-0.5, 8, ngrid)
    w2range = np.linspace(-1.5, 1.5, ngrid)
    w1,w2 = np.meshgrid(w1range,w2range)

    cost = np.array([[np.log(((x.dot(np.array([w1i,w2j]))-Y)**2).sum()) for w1i in w1range] for  w2j in w2range])

    plt.figure()
    plt.contour(w1, w2, cost)
    plt.scatter(wstar[0], wstar[1],c='r')
    plt.plot(allw[:,0],allw[:,1],'b+-' ,lw=2 )
    plt.show()

def mc2(X2,Y2):
    c1 = (X2*X2).reshape(N,1)
    c2 = X2.reshape(N,1)
    c3 = np.ones((N,1))
    X = np.hstack((c1,c2,c3))
    print('X : ', X)

def test_methode(X,Y, f):
    a, b  = f(X,Y)
    model = [ a* xi + b for xi in X]
    Mdraw(X,Y,model,f.__name__)
    

#============================================================
if __name__ == '__main__':
    a = 6.
    b = -1.
    N = 100
    sig = .4 # écart type
    X,Y = toy_data(a,b,N,sig)
    N = len(X)
    x = np.hstack((X.reshape(N,1),np.ones((N,1))))
    y = np.hstack((Y.reshape(N,1),np.ones((N,1))))
    wstar = np.linalg.solve(x.T.dot(x), x.T.dot(y))
    #test_methode(X,Y, moindre_carre)
    #test_methode(X,Y, estimate_parameter)
    print('x :' ,x )
    print('y :',y)
    a, b, allw = descente_gradient(x,Y)
    model = [ a* xi + b for xi in X]
    Mdraw(X,Y,model,'descente_gradient')
    gradient_draw()

    X2, Y2 = toy_data2(a,b,0,N,sig)
    draw(X2, Y2)
    mc2(X2, Y2)

