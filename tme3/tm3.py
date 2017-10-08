# -*- coding: utf-8 -*- 

import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D

def read_file ( filename ):
    """
    Lit un fichier USPS et renvoie un tableau de tableaux d'images.
    Chaque image est un tableau de nombres réels.
    Chaque tableau d'images contient des images de la même classe.
    Ainsi, T = read_file ( "fichier" ) est tel que T[0] est le tableau
    des images de la classe 0, T[1] contient celui des images de la classe 1,
    et ainsi de suite.
    """
    # lecture de l'en-tête
    infile = open ( filename, "r" )    
    nb_classes, nb_features = [ int( x ) for x in infile.readline().split() ]

    # creation de la structure de données pour sauver les images :
    # c'est un tableau de listes (1 par classe)
    data = np.empty ( 10, dtype=object )   
    filler = np.frompyfunc(lambda x: list(), 1, 1)
    filler( data, data )

    # lecture des images du fichier et tri, classe par classe
    for ligne in infile:
        champs = ligne.split ()
        if len ( champs ) == nb_features + 1:
            classe = int ( champs.pop ( 0 ) )
            data[classe].append ( list ( map ( lambda x: float(x), champs ) ) )
    infile.close ()

    # transformation des list en array
    output  = np.empty ( 10, dtype=object )
    filler2 = np.frompyfunc(lambda x: np.asarray (x), 1, 1)
    filler2 ( data, output )

    return output

def display_image ( X ):
    """
    Etant donné un tableau X de 256 flotants représentant une image de 16x16
    pixels, la fonction affiche cette image dans une fenêtre.
    """
    # on teste que le tableau contient bien 256 valeurs
    if X.size != 256:
        raise ValueError ( "Les images doivent être de 16x16 pixels" )

    # on crée une image pour imshow: chaque pixel est un tableau à 3 valeurs
    # (1 pour chaque canal R,G,B). Ces valeurs sont entre 0 et 1
    Y = X / X.max ()
    img = np.zeros ( ( Y.size, 3 ) )
    for i in range ( 3 ):
        img[:,i] = X

    # on indique que toutes les images sont de 16x16 pixels
    img.shape = (16,16,3)

    # affichage de l'image
    plt.imshow( img )
    plt.show ()

def learnML_class_parameters(tabImage):
    """float np.array np.array -> float np.array x float np.array """
    array_ui = np.zeros(256)
    array_sigma = np.zeros(256)
    #calcul de la moyenne
    for image in tabImage:
        for i in range(len(image)):
            pixel = image[i]
            array_ui[i] += pixel
    #calcul de l'estimateur de la variance
    array_ui /= len(tabImage)
    for image in tabImage:
        for i in range(len(image)):
            pixel = image[i]
            array_sigma[i] += (pixel - array_ui[i])**2
    #Selon les valeurs du TP il faut avoir s² = SIGMA(xi - m)² / n alors que la variance corrigé utilise n-1 au denominateur pourquoi ?
    array_sigma /= len(tabImage)
    return array_ui, array_sigma

def learnML_all_parameters(MatriceImage):
    """float np.array np.array np.array -> (float np.array x float np.array) list"""
    L = []
    for tabImage in MatriceImage:
        array_ui, array_sigma = learnML_class_parameters(tabImage)
        L.append((array_ui, array_sigma))
    return L

def log_likelihood(image, tupleParameters):
    """ float np.array x (float np.array,np.array) -> float """
    s = 0
    u, sigma = tupleParameters
    for i in range(len(image)):
        pixel = image[i]
        ui = u[i]
        si = sigma[i]
        if si != 0:
            s += -(1.0/2.0)*math.log(2*math.pi*si) -(1.0/2.0)*((pixel - ui)**2 / (si))
    return s

def log_likelihoods(image, parameters):
    """float np.array x (float np.array,np.array) list -> float np.array """
    return  [ log_likelihood (image, parameters[i] ) for i in range ( 10 ) ]

def classify_image(image, parameters):
    """float np.array x (float np.array,np.array) list -> int"""
    L = log_likelihoods(image, parameters)
    maxi = -1*float("inf")
    indexMax = None
    for i in range(len(L)):
        log_vraisemblance = L[i]
        if log_vraisemblance > maxi:
            maxi = log_vraisemblance
            indexMax = i
    return indexMax

def classify_all_image(MatriceImage, parameters):
    """float np.array np.array np.array x (np.array,np.array) list -> float np.2D-array """
    T = np.zeros((10,10))
    for label in range(len(MatriceImage)):
        TabImage = MatriceImage[label]
        for image in TabImage:
            predict = classify_image(image, parameters)
            T[label][predict] += 1
        #transformation proba
        T[label] /= len(TabImage)
    return T

def test_classify_all_image(MatriceImage, parameters, epsilon=0.0001):
    T = classify_all_image(MatriceImage, parameters)
    for label in T:
        s = 0
        for proba in label:
            s += proba
        if s > 1 + epsilon or s < 1 - epsilon:
            print('La proba ne vaut pas 1, s : ', s)
    print(T[0,0])
    print(T[2,3])
    print(T[5,3])

def dessine ( classified_matrix ):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.linspace ( 0, 9, 10 )
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, classified_matrix, rstride = 1, cstride=1 )
    plt.show()

def main():
    training_data = read_file("2015_tme3_usps_train.txt")
    #display_image( training_data[2][0])
    #display_image(training_data[3][4])
    parameters = learnML_all_parameters(training_data)
    test_data = read_file("2015_tme3_usps_test.txt")
    #l = log_likelihood(test_data[2][3], parameters[1])
    #test_classify_all_image(test_data, parameters)
    T = classify_all_image(test_data, parameters)
    dessine(T)
    

main()
