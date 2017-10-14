# -*- coding: utf-8 -*-

import numpy as np
import math
from pylab import *
import matplotlib.pyplot as plt

def read_file ( filename ):
    """
    Lit le fichier contenant les données du geyser Old Faithful
    """
    # lecture de l'en-tête
    infile = open ( filename, "r" )
    for ligne in infile:
        if ligne.find ( "eruptions waiting" ) != -1:
            break

    # ici, on a la liste des temps d'éruption et des délais d'irruptions
    data = []
    for ligne in infile:
        nb_ligne, eruption, waiting = [ float (x) for x in ligne.split () ]
        data.append ( eruption )
        data.append ( waiting )
    infile.close ()

    # transformation de la liste en tableau 2D
    data = np.asarray ( data )
    data.shape = ( int ( data.size / 2 ), 2 )

    return data

def dessine_1_normale ( params ):
    # récupération des paramètres
    mu_x, mu_z, sigma_x, sigma_z, rho = params

    # on détermine les coordonnées des coins de la figure
    x_min = mu_x - 2 * sigma_x
    x_max = mu_x + 2 * sigma_x
    z_min = mu_z - 2 * sigma_z
    z_max = mu_z + 2 * sigma_z

    # création de la grille
    x = np.linspace ( x_min, x_max, 100 )
    z = np.linspace ( z_min, z_max, 100 )
    X, Z = np.meshgrid(x, z)

    # calcul des normales
    norm = X.copy ()
    for i in range ( x.shape[0] ):
        for j in range ( z.shape[0] ):
            norm[i,j] = normale_bidim ( x[i], z[j], params )

    # affichage
    fig = plt.figure ()
    plt.contour ( X, Z, norm, cmap=cm.autumn )
    plt.show ()

def dessine_normales ( data, params, weights, bounds, ax ):
    # récupération des paramètres
    mu_x0, mu_z0, sigma_x0, sigma_z0, rho0 = params[0]
    mu_x1, mu_z1, sigma_x1, sigma_z1, rho1 = params[1]

    # on détermine les coordonnées des coins de la figure
    x_min = bounds[0]
    x_max = bounds[1]
    z_min = bounds[2]
    z_max = bounds[3]

    # création de la grille
    nb_x = nb_z = 100
    x = np.linspace ( x_min, x_max, nb_x )
    z = np.linspace ( z_min, z_max, nb_z )
    X, Z = np.meshgrid(x, z)

    # calcul des normales
    norm0 = np.zeros ( (nb_x,nb_z) )
    for j in range ( nb_z ):
        for i in range ( nb_x ):
            norm0[j,i] = normale_bidim ( x[i], z[j], params[0] )# * weights[0]
    norm1 = np.zeros ( (nb_x,nb_z) )
    for j in range ( nb_z ):
        for i in range ( nb_x ):
             norm1[j,i] = normale_bidim ( x[i], z[j], params[1] )# * weights[1]

    # affichages des normales et des points du dataset
    ax.contour ( X, Z, norm0, cmap=cm.winter, alpha = 0.5 )
    ax.contour ( X, Z, norm1, cmap=cm.autumn, alpha = 0.5 )
    for point in data:
        ax.plot ( point[0], point[1], 'k+' )


def find_bounds ( data, params ):
    # récupération des paramètres
    mu_x0, mu_z0, sigma_x0, sigma_z0, rho0 = params[0]
    mu_x1, mu_z1, sigma_x1, sigma_z1, rho1 = params[1]

    # calcul des coins
    x_min = min ( mu_x0 - 2 * sigma_x0, mu_x1 - 2 * sigma_x1, data[:,0].min() )
    x_max = max ( mu_x0 + 2 * sigma_x0, mu_x1 + 2 * sigma_x1, data[:,0].max() )
    z_min = min ( mu_z0 - 2 * sigma_z0, mu_z1 - 2 * sigma_z1, data[:,1].min() )
    z_max = max ( mu_z0 + 2 * sigma_z0, mu_z1 + 2 * sigma_z1, data[:,1].max() )

    return ( x_min, x_max, z_min, z_max )


# -------------------------------------------------------------------------
# FIN DES FONCTION DU PROF

def normale_bidim(x, z, t):
    """float x float x (float,float,float,float,float) -> float """
    uX, uZ, sigmaX, sigmaZ, p = t
    g = 1.0 / (1.0*(2*math.pi*sigmaX*sigmaZ*math.sqrt(1 - p**2)))
    d = ((x - uX) / (sigmaX * 1.0))**2 - 2*p* ((x-uX)*(z-uZ)/ (sigmaX * sigmaZ * 1.0)) + ((z - uZ) / (1.0 * sigmaZ))**2
    mult = (- 1.0 / (2 *(1-p**2))) *d
    return g * math.exp(mult)

def Q_i(data, current_params, current_weights):
    """ float np.2D-array x float np.2D-array x float np.array -> float np.2D-array """
    Q = []
    for point in data:
        alpha0 = current_weights[0] * normale_bidim(point[0], point[1], current_params[0])
        alpha1 = current_weights[1] * normale_bidim(point[0], point[1], current_params[1])
        Q0 =  1.0 * alpha0 / (1.0*(alpha0 + alpha1))
        Q1 = 1.0 * alpha1 / (1.0*(alpha0 + alpha1))
        Q.append([Q0, Q1])
    return np.array(Q)

def M_step(data, T, current_params, current_weights):
    """ np.2D-array x np.2D-array x float np.2D-array x float np.array -> float np.2D-array x float np.array """
    pi = []
    params0 = []
    params1 = []
    Tsum = T.sum()
    print(T[:,0])
    Pi0 = T[:,0].sum() / (1.0 * Tsum)
    Pi1 = T[:,1].sum() / (1.0 * Tsum)
    pi.append((Pi0, Pi1))
    Q0 = T[:, 0]
    Q1 = T[:, 1]
    Q0sum = Q0.sum()
    Q1sum = Q1.sum()
    #calcul des params des u
    sx0= 0
    sz0 = 0
    sx1 = 0
    sz1 = 0
    for i in range(len(Q0)):
        point = data[i]
        Qi0 = Q0[i]
        Qi1 = Q1[i]
        sx0 += Qi0 * point[0]
        sz0 += Qi0 * point[1]
        sx1 += Qi1 * point[0]
        sz1 += Qi1 * point[1]
        
    UX0 = sx0 / (1.0*(T[:,0].sum()))
    UZ0 = sz0 / (1.0*(T[:,0].sum()))

    params0.append(UX0)
    params0.append(UZ0)

    UX1 = sx1 / ((1.0*(Q1.sum())))
    UZ1 = sz1 / ((1.0*(Q1.sum())))

    params1.append(UX1)
    params1.append(UZ1)

    siX0 = 0
    siX1 = 0
    siZ0 = 0
    siZ1 = 0
    for i in range(len(Q0)):
        point = data[i]
        Qi0 = Q0[i]
        Qi1 = Q1[i]
        siX0 += Qi0 * ((point[0] - UX0)**2)
        siZ0 += Qi0 * ((point[1] - UZ0)**2)
        siX1 += Qi1 * ((point[0] - UX1)**2)
        siZ1 += Qi1 * ((point[1] - UZ1)**2)    

    sigmaX0 = math.sqrt(siX0 / (1.0 * Q0sum))
    sigmaZ0 = math.sqrt(siZ0 / (1.0 * Q0sum))
    sigmaX1 = math.sqrt(siX1 / (1.0 * Q1sum))
    sigmaZ1 = math.sqrt(siZ1 / (1.0 * Q1sum))    
    params0.append(sigmaX0)
    params0.append(sigmaZ0)
    params1.append(sigmaX1)
    params1.append(sigmaZ1)

    for i in range(len(Q0)):
        point = data[i]
        Qi0 = Q0[i]
        Qi1 = Q1[i]
    
    #print('{} {}'.format(Pi0, Pi1))

def affiche_volcan():
    data = read_file ( "2015_tme4_faithful.txt" )
    print('data : ', data[0])
    # affichage des données : calcul des moyennes et variances des 2 colonnes
    mean1 = data[:,0].mean()
    mean2 = data[:,1].mean()
    std1  = data[:,0].std()
    std2  = data[:,1].std()
    print('m1 : {}, m2 : {}, std1 : {}, std2 : {}'.format(mean1, mean2, std1, std2))

    # les paramètres des 2 normales sont autour de ces moyennes
    params = np.array ( [(mean1 - 0.2, mean2 - 1, std1, std2, 0),
                         (mean1 + 0.2, mean2 + 1, std1, std2, 0)] )
    weights = np.array ( [0.4, 0.6] )
    bounds = find_bounds ( data, params )

    # affichage de la figure
    fig = plt.figure ()
    ax = fig.add_subplot(111)
    dessine_normales ( data, params, weights, bounds, ax )
    #affichage incorect, il doit y avoir un probleme dans les parametres des gaussiennes
    plt.show ()


def main():
    data = read_file ( "2015_tme4_faithful.txt" )
    current_params = array([(2.51460515, 60.12832316, 0.90428702, 11.66108819, 0.86533355),
                        (4.2893485,  79.76680985, 0.52047055,  7.04450242, 0.58358284)])
    current_weights = array([ 0.45165145,  0.54834855])
    Q = Q_i ( data, current_params, current_weights )
    M_step(data, Q, current_params, current_weights)
    #print(T)

main()
