# -*- coding: utf-8 -*-

import numpy as np

# fonction pour transformer les données brutes en nombres de 0 à n-1
def translate_data ( data ):
    # création des structures de données à retourner
    nb_variables = data.shape[0]
    nb_observations = data.shape[1] - 1 # - nom variable
    res_data = np.zeros ( (nb_variables, nb_observations ), int )
    res_dico = np.empty ( nb_variables, dtype=object )

    # pour chaque variable, faire la traduction
    for i in range ( nb_variables ):
        res_dico[i] = {}
        index = 0
        for j in range ( 1, nb_observations + 1 ):
            # si l'observation n'existe pas dans le dictionnaire, la rajouter
            if data[i,j] not in res_dico[i]:
                res_dico[i].update ( { data[i,j] : index } )
                index += 1
            # rajouter la traduction dans le tableau de données à retourner
            res_data[i,j-1] = res_dico[i][data[i,j]]
    return ( res_data, res_dico )


# fonction pour lire les données de la base d'apprentissage
def read_csv ( filename ):
    data = np.loadtxt ( filename, delimiter=',', dtype=np.str ).T
    names = data[:,0].copy ()
    data, dico = translate_data ( data )
    return names, data, dico

# etant donné une BD data et son dictionnaire, cette fonction crée le
# tableau de contingence de (x,y) | z
def create_contingency_table ( data, dico, x, y, z ):
    # détermination de la taille de z
    size_z = 1
    offset_z = np.zeros ( len ( z ) )
    j = 0
    for i in z:
        offset_z[j] = size_z      
        size_z *= len ( dico[i] )
        j += 1

    # création du tableau de contingence
    res = np.zeros ( size_z, dtype = object )

    # remplissage du tableau de contingence
    if size_z != 1:
        z_values = np.apply_along_axis ( lambda val_z : val_z.dot ( offset_z ),
                                         1, data[z,:].T )
        i = 0
        while i < size_z:
            indices, = np.where ( z_values == i )
            a,b,c = np.histogram2d ( data[x,indices], data[y,indices],
                                     bins = [ len ( dico[x] ), len (dico[y] ) ] )
            res[i] = ( indices.size, a )
            i += 1
    else:
        a,b,c = np.histogram2d ( data[x,:], data[y,:],
                                 bins = [ len ( dico[x] ), len (dico[y] ) ] )
        res[0] = ( data.shape[1], a )
    return res

def sufficient_statistics(data, dico, x, y, z):
    """ int np.2D-array x dico{string -> int} np.array x int x int x int list -> float """
    pass


def main(v=False):
    # names : tableau contenant les noms des variables aléatoires
    # data  : tableau 2D contenant les instanciations des variables aléatoires
    # dico  : tableau de dictionnaires contenant la correspondance (valeur de variable -> nombre)
    #names, data, dico = read_csv ( "2015_tme5_asia.csv" )
    names, data, dico = read_csv('test.txt')
    print(data)
    print('----------------------')
    #bug
    v1 = data[:,0]
    v2 = data[:,2]
    print(v1)
    print(v2)
    print('---------------------')
    r = np.vstack((v1,v2))
    print(r.T)
    resultat = create_contingency_table(r.T, dico, 0, 2, [3])
    if v:
        print('names : ', names)
        print('data : ', data)
        print('dico : ', dico)
        print('resultat', resultat)

main(True)
