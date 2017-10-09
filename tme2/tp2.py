# coding: utf-8

import random
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyAgrum as gum
import pyAgrum.lib.ipython as gnb
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
    """ float np.array x float np.array -> float np.2D-array 
    on fait le produit tensoriel et non le produit matriciel"""
    return np.outer(x,y)

def produit_tensoriel(A, B):
    """ A : matrice, B : matrice """
    M = []
    s1 = np.shape(A)
    s2 = np.shape(B)
    #A is 1 d
    L_ai = []
    L_bi = []
    if len(s1) == 1:
        L_ai = A
    else:
        for l in A:
            for ai in l:
                L_ai.append(ai)
    if len(s2) == 1:
        L_bi = B
    else:
        for l in B:
            for bi in l:
                L_bi.append(bi)

    for ai in L_ai:
        L = []
        for bi in L_bi:
            L.append(ai * bi)
        M.append(L)
    return np.array(M)
        
    return np.array(M)
        

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

def independance_conditionnelles():
    P_XYZT = np.array([[[[ 0.0192,  0.1728],
                     [ 0.0384,  0.0096]],

                    [[ 0.0768,  0.0512],
                     [ 0.016 ,  0.016 ]]],

                   [[[ 0.0144,  0.1296],
                     [ 0.0288,  0.0072]],

                    [[ 0.2016,  0.1344],
                     [ 0.042 ,  0.042 ]]]])

    P_YZ = np.zeros((2,2))
    for zi in range(len(P_XYZT)):
        for yi in range(len(P_XYZT)):
            s = 0
            for xi in range(len(P_XYZT)):
                for ti in range(len(P_XYZT)):
                    s += P_XYZT[xi][yi][zi][ti]
            P_YZ[yi][zi] = s
    print(P_YZ)

    P_XTcondYZ = np.zeros((2,2,2,2))
    for xi in range(len(P_XYZT)):
        for yi in range(len(P_XYZT)):
            for zi in range(len(P_XYZT)):
                for ti in range(len(P_XYZT)):
                    P_XTcondYZ[xi][yi][zi][ti] = P_XYZT[xi][yi][zi][ti] / P_YZ[yi][zi]
    print('P_XTcondYZ')
    print(P_XTcondYZ)

    print('P_XcondYZ')
    P_XcondYZ = np.zeros((2,2,2))
    for xi in range(len(P_XYZT)):
        for yi in range(len(P_XYZT)):
            for zi in range(len(P_XYZT)):
                s = 0
                for ti in range(len(P_XYZT)):
                    s += P_XTcondYZ[xi][yi][zi][ti]
                P_XcondYZ[xi][yi][zi] = s
    print(P_XcondYZ)

    print('P_TcondYZ')
    P_TcondYZ = np.zeros((2,2,2))
    for ti in range(len(P_XYZT)):
        for yi in range(len(P_XYZT)):
            for zi in range(len(P_XYZT)):
                s = 0
                for xi in range(len(P_XYZT)):
                    s += P_XTcondYZ[xi][yi][zi][ti]
                P_TcondYZ[ti][yi][zi] = s
    print(P_TcondYZ)

    print('Verif independance')
    print(np.outer(P_XcondYZ, P_TcondYZ))

    P_XYZ = np.zeros((2,2,2))
    for xi in range(len(P_XYZT)):
        for yi in range(len(P_XYZT)):
            for zi in range(len(P_XYZT)):
                s = 0
                for ti in range(len(P_XYZT)):
                    s += P_XYZT[xi][yi][zi][ti]
                P_XYZ[xi][yi][zi] = s
    print('P_XYZ')
    print(P_XYZ)

    P_X = np.zeros((2))
    for xi in range(len(P_XYZ)):
        s = 0
        for yi in range(len(P_XYZ)):
            for zi in range(len(P_XYZ)):
                s += P_XYZ[xi][yi][zi]
        P_X[xi] = s
    print('Px')
    print(P_X)

    P_YZ = np.zeros((2,2))
    for yi in range(len(P_XYZ)):
        for zi in range(len(P_XYZ)):
            s = 0
            for xi in range(len(P_XYZ)):
                s += P_XYZ[xi][yi][zi]
            P_YZ[yi][zi] = s
    print('P_YZ')
    print(P_YZ)

    print("P(X) * P(Y,Z)")
    M = np.outer(P_X, P_YZ)
    print(M)
    #print(np.isclose(M,P_XYZ))

def read_file ( filename ):
    """
    Renvoie les variables aléatoires et la probabilité contenues dans le
    fichier dont le nom est passé en argument.
    """
    Pjointe = gum.Potential ()
    variables = []

    fic = open ( filename, 'r' )
    # on rajoute les variables dans le potentiel
    nb_vars = int ( fic.readline () )
    for i in range ( nb_vars ):
        name, domsize = fic.readline ().split ()
        variable = gum.LabelizedVariable(name,name,int (domsize))
        variables.append ( variable )
        Pjointe.add(variable)

    # on rajoute les valeurs de proba dans le potentiel
    cpt = []
    for line in fic:
        cpt.append ( float(line) )
    Pjointe.fillWith(np.array ( cpt ) )

    fic.close ()
    return np.array ( variables ), Pjointe

def conditional_indep(potential, X, Y, Z, epsilon):
    PXYZ = potential.margSumIn([x.name() for x in [X,Y] + Z])
    P_XZ = PXYZ.margSumOut([Y.name()])
    P_Z = PXYZ.margSumOut([Y.name(), X.name()])
    P_XsachantZ = P_XZ / P_Z
    P_YZ = PXYZ.margSumOut([X.name()])
    P_YsachantZ = P_YZ / P_Z
    Q = PXYZ - (P_XsachantZ * P_YsachantZ)
    v = Q.toarray()
    b = abs(v) > epsilon
    return not(np.any(b))

def compact_conditional_proba(potential, X, epsilon):
    #K = toutes les variables sauf X
    K = potential.margSumIn(X.name())
    for xi in potential.variableSequence():
        if conditional_indep(potential, xi, X, K , epsilon):
            K -= xi
    return potential / k

def create_bayesian_network(proba_jointe, epsilon):
    liste = {}
    #copy
    P = proba_jointe
    print(P)
    #bug var est un array de valeur comment parcourir les noms de variables ?
    for var in proba_jointe.variableSequence():
        print('var : ', var)
        Q = compact_conditional_proba(P, var.name(), epsilon)
        liste.add(Q)
        print('name : ', var.name)
        P = P.margSumIm([var.name()])
    return liste

def test_bayesien():
    LabelizedVariable, potential = read_file('asia.txt')
    b = create_bayesian_network(potential, 0.01)
    gnb.showPotential(b)
    

def gum_part():
    LabelizedVariable, potential = read_file('asia.txt')
    #print(LabelizedVariable)
    #print(potential)
    Q =conditional_indep(potential, LabelizedVariable[0], LabelizedVariable[1], [LabelizedVariable[2]], 0.1)
    print(Q)
    
# -----------------------------------------------------
#                 TEST
# -----------------------------------------------------
    
def test_Pjointe():
    PA = np.array ( [0.2, 0.7, 0.1] )
    PB = np.array ( [0.4, 0.4, 0.2] )
    Pjointe = Pxy(PA, PB)
    dessine(Pjointe)

def test_Pjointe_normal():
    n1 = np.array(normale(21,1)[0])
    n2 = np.array(normale(21,1)[0])
    Pjointe = Pxy(n1,n2)
    dessine(Pjointe)
    
def main():
    #histogramme_binomiale(20)
    #plot_normale(1001,1)
    #plot_affine(21,0.001)
    #independance_conditionnelles()
    #gum_part()
    test_bayesien()
    

main()
