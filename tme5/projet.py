# -*- coding: utf-8 -*-

import pydot        
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy.stats as stats
import pyAgrum as gum
import pyAgrum.lib.ipython as gnb

style = { "bgcolor" : "#6b85d1", "fgcolor" : "#FFFFFF" }

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

def display_BN ( node_names, bn_struct, bn_name, style ):
    graph = pydot.Dot( bn_name, graph_type='digraph')

    # création des noeuds du réseau
    for name in node_names:
        new_node = pydot.Node( name, 
                               style="filled",
                               fillcolor=style["bgcolor"],
                               fontcolor=style["fgcolor"] )
        graph.add_node( new_node )

    # création des arcs
    for node in range ( len ( node_names ) ):
        parents = bn_struct[node]
        for par in parents:
            new_edge = pydot.Edge ( node_names[par], node_names[node] )
            graph.add_edge ( new_edge )

    # sauvegarde et affaichage
    outfile = bn_name + '.png'
    graph.write_png( outfile )
    img = mpimg.imread ( outfile )
    plt.imshow( img )

def learn_parameters ( bn_struct, ficname ):
    # création du dag correspondant au bn_struct
    graphe = gum.DAG ()
    nodes = [ graphe.addNode () for i in range ( bn_struct.shape[0] ) ]
    for i in range ( bn_struct.shape[0] ):
        for parent in bn_struct[i]:
            graphe.addArc ( nodes[parent], nodes[i] )

    # appel au BNLearner pour apprendre les paramètres
    learner = gum.BNLearner ( ficname )
    learner.useScoreLog2Likelihood ()
    learner.useAprioriSmoothing ()
    return learner.learnParameters ( graphe )

#----------------------------------------------------------------------
#                     MES FONCTIONS
#----------------------------------------------------------------------

def calculate_Nxz(resultat, v=False):
        M = []
        L = []
        for Rz in resultat:
            L = []
            Txy = Rz[1]
            #on parcours les lignes
            for line in Txy:
                if v:
                    print('line : ', line)
                L.append(line.sum())
            M.append(L)
        Nxz = np.array(M).T
        return Nxz

def calculate_Nyz(resultat, v=False):
    M = []
    for Rz in resultat:
        L = []
        Txy = Rz[1]
        if v:
            print(Txy)
        t = Txy.shape
        if v:
            print(t)
        #on parcours les colonnes
        for j in range(t[1]):
            col = Txy[:,j]
            if v:
                print('col',col)
            L.append(col.sum())
        M.append(L)
    Nyz = np.array(M).T
    return Nyz

def correct_variable(x):
    if x == 0:
        return 1
    return x

def sufficient_statistics(data, dico, x, y, z):
    """ int np.2D-array x dico{string -> int} np.array x int x int x int list -> float """
    #on construit la statistique d'ajustement grace aux tableau de contingence
    #Ici la loi à vérifier est les données sont indépendantes : donc n*pr = Nz * Ny/ N
    resultat = create_contingency_table(data, dico, x, y, z)
    test = 0

    N_xz = calculate_Nxz(resultat)
    N_yz = calculate_Nyz(resultat)
    cptZ = 0

    z = 0
    for Rz in resultat:
        #print('z : ', z)
        NZ = Rz[0]
        if NZ != 0:
            cptZ += 1
        Txy = Rz[1]
        shape = Txy.shape
        #print('shape : ', shape)
        for xi in range(shape[0]):
            for yi in range(shape[1]):
                XYZ = Txy[xi][yi]
                NXZ = N_xz[xi][z]
                NYZ = N_yz[yi][z]
                #Au lieu de ne pas prendre les cas ou la valeur est 0, on peut remplacer la valeur génante par 1
                #if NZ == 0 and NXZ == 0 and NYZ == 0:
                #    test += (XYZ)**2
                #else:
                #    NZ = correct_variable(NZ)
                #    NXZ = correct_variable(NXZ)
                #    NYZ = correct_variable(NYZ)
                #vu les données de l'énoncé il faut faire comme ça
                if NZ != 0 and NXZ != 0 and NYZ != 0:
                    test += (XYZ - ((NXZ * NYZ) / NZ))**2 / ((NXZ * NYZ) / NZ)
        z += 1
    deg_liberte = cptZ * (len(dico[x]) - 1) * (len(dico[y]) - 1)
    return test, deg_liberte

def indep_score(data, dico, x, y, z):
    """ int np.2D-array x dico{string -> int} np.array x int x int x int list -> (float,int) """
    loi, d = sufficient_statistics(data, dico, x, y, z)
    resultat = create_contingency_table(data, dico, x, y, z)
    Z = len(resultat) # ???
    dmin = 5 * len(dico[x]) * len(dico[y]) * Z
    if len(data[0]) < dmin:
        return (-1,1)
    else:
        return (stats.chi2.sf(loi, d), d)

def best_candidate(data, dico, index_X, L_Z, alpha, debug=False):
    """int np.2D-array x dico{string -> int} np.array x int x int list x float -> int"""
    min_candidat = float("inf")
    index_candidat = None
    #on calcule les yi d'index < à X
    for yi in range(index_X):
        p_value,d  = indep_score(data, dico, index_X, yi, L_Z)
        if p_value < min_candidat and p_value < alpha:
            if debug:
                print("p_value : ", p_value)
                print("alpha : ", alpha)
                print("index : ", yi)
            min_candidat = p_value
            index_candidat = yi
    L = []
    if index_candidat != None:
        L.append(index_candidat)
    return L

def create_parents(data, dico, x, alpha,debug=False):
    z = []
    r = best_candidate(data, dico , x, z, alpha)
    cpt = 0
    while r != []:
        z += r
        r = best_candidate(data, dico , x, z, alpha)
        if debug:
            print('z : ', z)
            print('r : ', r)
            print('cpt : ', cpt)
        cpt += 1
    return z

def learn_BN_structure(data, dico, alpha):
    tab = []
    for variable in range(len(data)):
        res = create_parents(data, dico, variable, alpha)
        tab.append(res)
    return np.array(tab)

#=====================================================
#               FONCTIONS DE TEST
#=====================================================

def test_best_candidate(data, dico):
    assert best_candidate ( data, dico, 1, [], 0.05 ) == []
    assert best_candidate ( data, dico, 4, [], 0.05 ) == [1]
    assert best_candidate ( data, dico, 4, [1], 0.05 ) == []
    assert best_candidate ( data, dico, 5, [], 0.05 ) == [3]
    assert best_candidate ( data, dico, 5, [6], 0.05 ) == [3]
    assert best_candidate ( data, dico, 5, [6,7], 0.05 ) == [2]

def test_create_parents(data, dico):
    assert create_parents ( data, dico, 1, 0.05 ) == []
    assert create_parents ( data, dico, 4, 0.05 ) == [1]
    assert create_parents ( data, dico, 5, 0.05 ) == [3,2]
    assert create_parents ( data, dico, 6, 0.05 ) == [4,5]    

def main(v=False):
    # names : tableau contenant les noms des variables aléatoires
    # data  : tableau 2D contenant les instanciations des variables aléatoires
    # dico  : tableau de dictionnaires contenant la correspondance (valeur de variable -> nombre)
    names, data, dico = read_csv ( "2015_tme5_asia.csv" )
    #names, data, dico = read_csv('test.txt')
    resultat = create_contingency_table(data, dico, 1, 2, [3])
    test_best_candidate(data, dico)
    test_create_parents(data, dico)
    bn_struct = learn_BN_structure(data, dico,0.05)
    bn = learn_parameters(bn_struct,"2015_tme5_asia.csv")
    print('taille : ', bn)
    gnb.showPotential( bn.cpt ( bn.idFromName ( 'bronchitis?' ) ) )
    proba = gum.getPosterior ( bn, {}, 'bronchitis?' )
    print('proba : ', proba)
    gnb.showPotential( proba )
    gnb.showPotential(gum.getPosterior ( bn,{'smoking?': 'true', 'tuberculosis?' : 'false' }, 'bronchitis?' ))
    style = { "bgcolor" : "#6b85d1", "fgcolor" : "#FFFFFF" }
    display_BN(names , bn_struct, 'BN', style)
    if v:
        print('names : ', names)
        print('data : ', data)
        print('dico : ', dico)
        print('resultat', resultat)

main()
