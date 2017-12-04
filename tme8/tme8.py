#coding: utf-8

import numpy as np
import pickle as pkl
from hmmlearn import hmm
import matplotlib.pyplot as plt

def observation_letter(Genome):
    dist = np.zeros((1,4))
    for letter in Genome:
        dist[0][letter] += 1
    return dist / len(Genome)

def build_bgene(Xgenes):
    """ on parcours la base de gene, on retire le premier et dernier codon et puis on parcours les codon en incrementant dist[position_Codon][lettre] """
    dist = np.zeros((3,4))
    for gene in Xgenes:
        for i in range(3, len(gene)-3):
            index = i % 3
            dist[index][gene[i]] += 1
    for line in dist:
        line /= np.sum(line)
    return dist

def build_Bstart():
    Bstart = np.array([[0.83,0,0.14,0.3],
                      [0,0,0,1],
                       [0,0,1,0]])
    return Bstart 

def build_bstop():
    Bstop = np.array([identite(3,4),identite(2,4), identite(0,4), identite(0,4), identite(2,4)])
    return Bstop
    
    

def draw(Annotation, pred):
    print('appel draw')
    Ltime = [i for i in range(6000)]
    plt.ylabel('Position du genome')
    plt.xlabel('Appartenance')
    plt.plot(Ltime, Annotation[:6000], "r.", label='Annotation')
    plt.plot(Ltime, pred[:6000], "g.", label='prediction')
    plt.legend()
    plt.show()

def multiple_draw(A1, pred1,pred2):
    print('appel multiple draw')
    Ltime = [i for i in range(6000)]
    plt.ylabel('Position du genome')
    plt.xlabel('Appartenance')
    plt.plot(Ltime, Annotation[:6000], "r.", label='Annotation')
    plt.plot(Ltime, pred[:6000], "g.", label='prediction model 1')
    plt.plot(Ltime, pred[:6000], "b.", label='prediction model 2')
    plt.legend()
    plt.show()

def identite(index, taille):
    A = [0 for i in range(taille)]
    A[index] = 1
    return A
            
        

with open('genome_genes.pkl', 'rb') as f:
        data = pkl.load(f)

Xgenes  = data.get("genes") #Les genes, une array de arrays

Genome = data.get("genome") #le premier million de bp de Coli

Annotation = data.get("annotation") ##l'annotation sur le genome
##0 = non codant, 1 = gene sur le brin positif

### Quelques constantes
DNA = ["A", "C", "G", "T"]
stop_codons = ["TAA", "TAG", "TGA"]


n_states_m1 = 4
# syntaxe objet python: créer un objet HMM
model1 = hmm.MultinomialHMM(n_components =  n_states_m1,init_params='ste')

#Sous question 1
#l'esperance de la loi géométrique est 1 / p donc pour p = a, on obtient que a = 1 / 200 bp
#Pour calculer b, on  prend la longueur moyenne d'un gène a priori
a = 1.0 / 200.0
s = 0
for gene in Xgenes:
    s += len(gene)
s = s / len(Xgenes)
#la taille moyenne d'un gène est de taille s. Par un raisonnement similaire que pour le paramètre a, on a b = 1 /s
b = 3.0 / s
print('a : ', a)
print('b : ', b)
#print('Xgenes : ', Xgenes[0])

Binter = observation_letter(Genome)
print('Binter :', Binter)

Bgene = build_bgene(Xgenes)
print('Bgene : ', Bgene)

Pi_m1 = np.array([1, 0, 0, 0]) ##on commence dans l'intergenique
A_m1 = np.array([[1-a, a  , 0, 0], 
                 [0  , 0  , 1, 0],
                 [0  , 0  , 0, 1],
                 [b  , 1-b, 0, 0 ]])

B_m1 = np.vstack((Binter, Bgene))

# paramétrage de l'objet
#model1._set_transmat(A_m1)
# [cf question d'après pour la détermination]
#model1._set_emissionprob(B_m1)
model1.startprob_ = Pi_m1
model1.transmat_ = (A_m1)
# [cf question d'après pour la détermination]
model1.emissionprob_ = (B_m1)

#la methode model1.decode demande un appel de la fonction fit
#fit sert à initialiser avant l'algorithme EM
#on transforme genome pour avoir le bon format
g = Genome.reshape(-1,1)
print('g : ' , g)
print('g shape' , g.shape)
#model1 = model1.fit(np.ndarray([1,0]))
vsbce, pred = model1.decode(g)
#print('pred :', pred)
#print('vsbce : ', vsbce)
sp = pred
sp[np.where(sp>=1)] = 1
#print('annotation : ', Annotation)
#print('sp : ', sp)
percpred1 = float(np.sum(sp == Annotation) )/ len(Annotation)
#draw(Annotation, pred)
print('accuracy model 1 : ', percpred1)

n2 = 12
Pi_m2 = np.array(identite(0,n2))
#print('Pi_m2 : ', Pi_m2)
A_m2 = np.array([[1-a,a,0,0,0,0,0,0,0,0,0,0], identite(2,n2), identite(3,n2), identite(4, n2), identite(5, n2), identite(6, n2),[0,0,0,0,1-b,0,0,b,0,0,0,0],[0,0,0,0,0,0,0,0,0.5,0.5,0,0], identite(10,n2), [0,0,0,0,0,0,0,0,0,0,0.5,0.5], identite(0, n2), identite(0,n2)])
B2inter = Binter.copy()
B2outer = Bgene.copy()
Bstart = build_Bstart()
Bstop = build_bstop()
B_m2 = np.vstack((Binter, Bstart,Bgene, Bstop))
print('A_m2 : ', A_m2)
for i in A_m2:
    print(len(i))
    
print('A_m1 shape : ', A_m1.shape)
print('A_m2 shape : ', A_m2.shape)
print('B_m2 shape : ', B_m2.shape)
print('B_m2', B_m2)
model2 = hmm.MultinomialHMM(n_components =  n2,init_params='ste')
model2.startprob_ = Pi_m2
model2.transmat_ = (A_m2)
# [cf question d'après pour la détermination]
model2.emissionprob_ = B_m2
vsbce, pred2 = model2.decode(g)
sp = pred2
sp[np.where(sp>=1)] = 1
percpred = float(np.sum(sp == Annotation) )/ len(Annotation)
#draw(Annotation, pred)
print('accuracy model 1 : ', percpred1)
assert(len(Pi_m2) == n2)
multiple_draw(Annotation, pred, pred2)
#assert(A_m2.shape == (n_m2, n_m2))
