import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

def dico0(t=100):
    d = dict()
    for i in range(t+1):
        d[i] = 0
    return d

def dico_add(dico, key):
    if key not in dico:
        dico[key] = 0
    else:
        dico[key] += 1
        
def improved_data():
    fname = "dataVelib.pkl"
    f = open(fname, "rb")
    data = pkl.load(f)
    f.close()
    print(len(data))
    L = []
    for station in data:
        s = []
        arrondissement = station['number'] // 1000
        if arrondissement > 0 or arrondissement < 21:
            s.append(station['position']['lat'])
            s.append(station['position']['lng'])
            s.append(station['alt'])
            s.append(arrondissement)
            s.append(station['bike_stands'])
            s.append(station['available_bike_stands'])
            L.append(s)
    new_data = np.array(L)
    return new_data

def histogramme(matrice, nItervalles):
    res = plt.hist(matrice, nItervalles)
    alt = res[1]
    intervalle = alt[1] - alt[0]
    pAlt = res[0]/res[0].sum()
    pAlt /= intervalle
    plt.bar((alt[1:]+alt[:-1])/2, pAlt, alt[1]-alt[0])
    plt.show()

def proba(matrice):
    total = matrice.sum()
    return matrice / total
    
def index_interval(LInterval, valeur):
    i = 0
    for v in LInterval:
        if valeur <= v:
            return i
        i += 1
    return i

def matrice_interval(LInterval, matrice):
    L = []
    for valeur in matrice:
        L.append(index_interval(LInterval, valeur))
    return np.array(L)

def build_Sp_Al(matrice):
    full = dico0()
    not_full = dico0()
    for line in matrice:
        if line[1] == line[2]:
            full[line[0]] += 1
        else:
            not_full[line[0]] += 1
    L = []
    for altitude in full.keys():
        #check 0
        deno = not_full[altitude] + full[altitude]
        if deno == 0:
            L.append([0,1])
        else:
            deno = (1.0 * (not_full[altitude] + full[altitude]))
            L.append([full[altitude] / deno, not_full[altitude] / deno])
    return (np.array(L))
                    

new_data = improved_data()
#print(new_data)
Ar = new_data[:,3]
print("Ar : ")
print(Ar)
P_Ar = proba(Ar)
print(P_Ar)
#check if Sum(P_Ar) = 1)
print(P_Ar.sum())
#print(mat_arrondissement)
mat_altitude = new_data[:,2]
#histogramme(mat_altitude, 100)
nItervalles = 100
res = plt.hist(mat_altitude, nItervalles)
Al = res[0]
print("Al : ")
print(Al)
print(len(Al))
P_Al = proba(Al)
 #check
print(P_Al.sum())
#------------------------ P[Sp|Al] ---------------
Altitude = matrice_interval(res[1], new_data[:, 2])
print("Altitude ")
print(Altitude)

mat_place_altitude = np.vstack((Altitude, new_data[:,4], new_data[:,5])).T
print("mat place altitude : ")
print(mat_place_altitude)
P_Sp_Al = build_Sp_Al(mat_place_altitude)
print("P_Sp_Al")
print(P_Sp_Al)
print(P_Al.sum())



