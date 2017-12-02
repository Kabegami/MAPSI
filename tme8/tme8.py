#coding: utf-8

import numpy as np
import pickle as pkl
from hmmlearn import hmm

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
            
        

with open('genome_genes.pkl', 'rb') as f:
        data = pkl.load(f, encoding='latin1')

Xgenes  = data.get("genes") #Les genes, une array de arrays

Genome = data.get("genome") #le premier million de bp de Coli

Annotation = data.get("annotation") ##l'annotation sur le genome
##0 = non codant, 1 = gene sur le brin positif

### Quelques constantes
DNA = ["A", "C", "G", "T"]
stop_codons = ["TAA", "TAG", "TGA"]


n_states_m1 = 4
# syntaxe objet python: créer un objet HMM
model1 = hmm.MultinomialHMM(n_components =  n_states_m1)

#Sous question 1
#l'esperance de la loi géométrique est 1 / p donc pour p = a, on obtient que a = 1 / 200 bp
#Pour calculer b, on  prend la longueur moyenne d'un gène a priori
a = 1.0 / 200.0
s = 0
for gene in Xgenes:
    s += len(gene)
s = s / len(Xgenes)
#la taille moyenne d'un gène est de taille s. Par un raisonnement similaire que pour le paramètre a, on a b = 1 /s
b = 1.0 / s
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
model1.startprob = Pi_m1
model1.transmat = (A_m1)
# [cf question d'après pour la détermination]
model1.emissionprob = (B_m1)

#la methode model1.decode demande un appel de la fonction fit
#fit sert à initialiser avant l'algorithme EM
#model1 = model1.fit(Genome.reshape(-1,1))
vsbce, pred = model1.decode(Genome)
sp = pred
sp[np.where(sp>=1)] = 1
percpred1 = float(np.sum(sp == Anotation ) / len(Annotation))
print('accuracy model 1 : ', percpred1)
