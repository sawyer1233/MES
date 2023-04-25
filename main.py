from math import sqrt
import numpy as np
import math
import scipy as sp

np.set_printoptions(edgeitems=30, linewidth=320)

filepath = "siatka2.txt"
file = open(filepath, "r")
content = file.readlines()
SimulationTime = content[0]
SimulationStepTime = content[1]
Conductivity = content[2]
Alfa = content[3]
Tot = content[4]
InitialTemp = content[5]
Density = content[6]
SpecificHeat = content[7]
Nodes_number = content[8]
Elements_number = content[9]
num1 = []
num2 = []
num3 = []
num4 = []
num5 = []
num6 = []
num7 = []
num8 = []
num9 = []
num10 = []
for numer in SimulationTime.split():
    if numer.isnumeric():
        num1.append(numer)

SimulationTimeint = int(num1[0])
print(SimulationTimeint)

for numer in SimulationStepTime.split():
    if numer.isnumeric():
        num2.append(numer)
SimulationStepTimeint = int(num2[0])
print(num2[0])
for numer in Conductivity.split():
    if numer.isnumeric():
        num3.append(numer)
Conductivityint = int(num3[0])
print(num3[0])
for numer in Alfa.split():
    if numer.isnumeric():
        num4.append(numer)
Alfaint = int(num4[0])
print(num4[0])
for numer in Tot.split():
    if numer.isnumeric():
        num5.append(numer)
Totint = int(num5[0])
print(num5[0])
for numer in InitialTemp.split():
    if numer.isnumeric():
        num6.append(numer)
InitialTempint = int(num6[0])
print(num6[0])
for numer in Density.split():
    if numer.isnumeric():
        num7.append(numer)
Densityint = int(num7[0])
print(num7[0])
for numer in SpecificHeat.split():
    if numer.isnumeric():
        num8.append(numer)
SpecificHeatint = int(num8[0])
print(num8[0])
for numer in Nodes_number.split():
    if numer.isnumeric():
        num9.append(numer)
Nodes_numberint = int(num9[0])
print(num9[0])
for numer in Elements_number.split():
    if numer.isnumeric():
        num10.append(numer)
Elements_numberint = int(num10[0])
print(num10[0])

punkty1 = content[11:27]
for line in punkty1:
    punkt, x, y = line.split(",")
    # print(punkt + "  x=" + x + "   y=" + y)
delim = ","
punkt, punkt_x, punkt_y = map(list, zip(*(ele.split(delim) for ele in punkty1)))

new_punkt_x = [float(x) for x in punkt_x]
new_punkt_y = [float(x) for x in punkt_y]
node_list = []
element_list = []

elementy = content[28:37]
elementy_id = []
elementy_node_id = []
for line in elementy:
    line = line.split(",")
    elementy_id.append(int(line[0]))
    pom = []
    pom.append(int(line[1]))
    pom.append(int(line[2]))
    pom.append(int(line[3]))
    pom.append(int(line[4]))
    elementy_node_id.append(pom)

#print(elementy_node_id)
# print(element_list)
BC_plik = content[38:39]
for x in BC_plik:
    BC = x.split(",")
BC = [int(i) for i in BC]


# print(new_punkt_x)

class Element:
    def __init__(self, ID, Node_id):
        self.ID = ID
        self.Node_Id = Node_id
        self.H = None
        self.C = None
        self.HBC = None
        self.P = None
        self.H_HBC = 0

class Node:
    def __init__(self, Id, x, y):
        self.Id = Id
        self.x = x
        self.y = y
        self.BC = None


for i in range(Nodes_numberint):
    node = Node(punkt[i], new_punkt_x[i], new_punkt_y[i])
    node_list.append(node)

for i in range(Elements_numberint):
    element = Element(elementy_id[i], elementy_node_id[i])
    element_list.append(element)
# print(node_list)
for node in node_list:
    if int(node.Id) in BC:
        node.BC = 1
    else:
        node.BC = 0


# do liczenia 9 punktów


def macierz(x0, x1, x2, x3, y0, y1, y2, y3):
    ksi = [-sqrt(3.0 / 5.0), 0, sqrt(3.0 / 5.0), -sqrt(3.0 / 5.0), 0, sqrt(3.0 / 5.0), -sqrt(3.0 / 5.0), 0,
           sqrt(3.0 / 5.0)]

    eta = [-sqrt(3.0 / 5.0), -sqrt(3.0 / 5.0), -sqrt(3.0 / 5.0), 0, 0, 0, sqrt(3.0 / 5.0), sqrt(3.0 / 5.0),
           sqrt(3.0 / 5.0)]

    wagi = [0 for x in range(9)]
    w1 = (5.0 / 9.0)
    w2 = 8.0 / 9.0
    w3 = 5.0 / 9.0
    wagi = [w1 * w1, w1 * w2, w1 * w3, w2 * w1, w2 * w2, w2 * w3, w3 * w1, w3 * w2, w3 * w3]
    tablica_dn_ksi = [[0 for x in range(4)] for y in range(9)]
    tablica_dn_eta = [[0 for x in range(4)] for y in range(9)]
    for j in range(9):
        tablica_dn_ksi[j][0] = -(0.25) * (1 - eta[j])
        tablica_dn_ksi[j][1] = (0.25) * (1 - eta[j])
        tablica_dn_ksi[j][2] = (0.25) * (1 + eta[j])
        tablica_dn_ksi[j][3] = -(0.25) * (1 + eta[j])
    # tablica_dn_ksi=np.array(tablica_dn_ksi)
    # print("Tablica ksi \n",tablica_dn_ksi)
    # for j in tablica_dn_ksi:
    #  print("\t".join(map(str,j)))
    for j in range(9):
        tablica_dn_eta[j][0] = -(0.25) * (1 - ksi[j])
        tablica_dn_eta[j][1] = -(0.25) * (1 + ksi[j])
        tablica_dn_eta[j][2] = (0.25) * (1 + ksi[j])
        tablica_dn_eta[j][3] = (0.25) * (1 - ksi[j])
    jakobi = [[[0 for x in range(2)] for y in range(2)] for z in range(9)]
    for j in range(9):
        jakobi[j][0][0] = tablica_dn_ksi[j][0] * x0 + tablica_dn_ksi[j][1] * x1 + tablica_dn_ksi[j][2] * x2 + \
                          tablica_dn_ksi[j][3] * x3
        jakobi[j][0][1] = tablica_dn_ksi[j][0] * y0 + tablica_dn_ksi[j][1] * y1 + tablica_dn_ksi[j][2] * y2 + \
                          tablica_dn_ksi[j][3] * y3
        jakobi[j][1][0] = tablica_dn_eta[j][0] * x0 + tablica_dn_eta[j][1] * x1 + tablica_dn_eta[j][2] * x2 + \
                          tablica_dn_eta[j][3] * x3
        jakobi[j][1][1] = tablica_dn_eta[j][0] * y0 + tablica_dn_eta[j][1] * y1 + tablica_dn_eta[j][2] * y2 + \
                          tablica_dn_eta[j][3] * y3
    odwjakobian = [0 for x in range(9)]
    jakobian = [0 for x in range(9)]
    for j in range(9):
        jakobian[j] = jakobi[j][1][1] * jakobi[j][0][0] - ((jakobi[j][0][1]) * (jakobi[j][1][0]))
        odwjakobian[j] = 1 / jakobian[j]
    tablica_dn_dx = [[0 for x in range(4)] for y in range(9)]
    tablica_dn_dy = [[0 for x in range(4)] for y in range(9)]
    for j in range(9):
        for k in range(4):
            tablica_dn_dx[j][k] = (odwjakobian[j] * jakobi[j][1][1] * tablica_dn_ksi[j][k]) + (
                    odwjakobian[j] * (-jakobi[j][0][1]) * tablica_dn_eta[j][k])
            tablica_dn_dy[j][k] = (odwjakobian[j] * (-jakobi[j][1][0]) * tablica_dn_ksi[j][k]) + (
                    odwjakobian[j] * jakobi[j][0][0] * tablica_dn_eta[j][k])
    jakobi = np.array(jakobi)

    # print(tablica_dn_dx)
    pom_x = [[[0 for x in range(4)] for y in range(4)] for z in range(9)]
    pom_y = [[[0 for x in range(4)] for y in range(4)] for z in range(9)]

    for j in range(9):
        for k in range(4):
            for l in range(4):
                pom_x[j][k][l] = tablica_dn_dx[j][k] * tablica_dn_dx[j][l]
                pom_y[j][k][l] = tablica_dn_dy[j][k] * tablica_dn_dy[j][l]
    macierz_h_ = [[0 for x in range(4)] for y in range(4)]
    macierz_c_dla_punktu = [[0 for x in range(4)] for y in range(4)]
    for j in range(9):
        for k in range(4):
            for l in range(4):
                macierz_h_[k][l] += (pom_x[j][k][l] + pom_y[j][k][
                    l]) * Conductivityint * jakobian[j] * wagi[j]


    #  print("\t".join(map(str,j)))

    #  funkcje ksztaltu
    N_tablica = [[0 for x in range(4)] for y in range(9)]

    for i in range(9):
        N_tablica[i][0] = 0.25 * (1 - ksi[i]) * (1 - eta[i])
        N_tablica[i][1] = 0.25 * (1 + ksi[i]) * (1 - eta[i])
        N_tablica[i][2] = 0.25 * (1 + ksi[i]) * (1 + eta[i])
        N_tablica[i][3] = 0.25 * (1 - ksi[i]) * (1 + eta[i])

    # liczenie macierzy c z numpy
    macierz_C = [[0 for x in range(4)] for y in range(4)]
    for i in range(9):
        macierz_N = np.array(N_tablica[i])
        macierz_N_transponowana = np.resize(macierz_N, (4, 1))
        macierz_N_mnozenie = macierz_N * macierz_N_transponowana
        macierz_N_mnozenie *= jakobian[i] * Densityint * SpecificHeatint * wagi[i]
        macierz_C += macierz_N_mnozenie

    MacierzDane = []
    MacierzDane.append(macierz_h_)
    MacierzDane.append(macierz_C)
    return MacierzDane


def agregateH():
    # agregacja H
    macierz_H_agregacja = [[0 for x in range(16)] for y in range(16)]
    for element in element_list:
        for i in range(4):
            for j in range(4):
                macierz_H_agregacja[element.Node_Id[i] - 1][element.Node_Id[j] - 1] += element.H[i][j]
    return macierz_H_agregacja


# agregacja C
def agregateC():
    macierz_C_agregacja = [[0 for x in range(16)] for y in range(16)]
    for element in element_list:
        for i in range(4):
            for j in range(4):
                macierz_C_agregacja[element.Node_Id[i] - 1][element.Node_Id[j] - 1] += element.C[i][j]
    return macierz_C_agregacja


def agregateP():
    wektorP = [[0 for x in range(1)] for y in range(16)]
    for element in element_list:
        for i in range(4):
            wektorP[element.Node_Id[i] - 1] += element.P[i]
    return wektorP
def agregate_h_hbc():
    macierz_h_hbc = [[0 for x in range(16)] for y in range(16)]
    for element in element_list:
        for i in range(4):
            for j in range(4):
                macierz_h_hbc[element.Node_Id[i] - 1][element.Node_Id[j] - 1] += element.H_HBC[i][j]
    return macierz_h_hbc

def f1(ksi, eta):
    return 0.25 * (1 - ksi) * (1 - eta)
def f2(ksi, eta):
    return 0.25 * (1 + ksi) * (1 - eta)
def f3(ksi, eta):
    return 0.25 * (1 + ksi) * (1 + eta)
def f4(ksi, eta):
    return 0.25 * (1 - ksi) * (1 + eta)

def calculateHbc(element_arg):

    PC1 = [(1, -sqrt(1 / 3)) ,(1, sqrt(1 / 3))]  # prawa
    PC2 = [(sqrt(1 / 3), 1) ,(-sqrt(1 / 3), 1)]  # gorna
    PC3 = [(-1, sqrt(1 / 3)) ,(-1, -sqrt(1 / 3))]  # lewa
    PC4 = [(-sqrt(1 / 3), -1) ,(sqrt(1 / 3), -1)]  # dolna
    wagi = [1,1]
    wymiar = (4,4)
    sciana_prawa = np.zeros(wymiar)
    sciana_gorna = np.zeros(wymiar)
    sciana_lewa = np.zeros(wymiar)
    sciana_dolna = np.zeros(wymiar)
    macierzHBC = np.zeros(wymiar)
    funkcja_ksztaltu = [0 for x in range(4)]
    detJ = 0
    for i in range(-1,3):
        x1 = node_list[element.Node_Id[i] - 1].x
        x2 = node_list[element.Node_Id[i + 1] - 1].x
        y1 = node_list[element.Node_Id[i] - 1].y
        y2 = node_list[element.Node_Id[i + 1] - 1].y
        if (element_arg.Node_Id[i] in BC) and (element_arg.Node_Id[i + 1] in BC):
            detJ = sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2)) / 2
          #  print("jakobian", detJ)
           # print(f"x1 = {x1}, x2 = {x2}, y1 = {y1}, y2 = {y2}")

            if i == -1:
                for j in range(2):
                    ksi = PC1[j][0]
                    eta = PC1[j][1]
                    funkcja_ksztaltu [0] = 0.25 * (1 + ksi) * (1 + eta)
                    funkcja_ksztaltu [1] = 0
                    funkcja_ksztaltu [2] = 0
                    funkcja_ksztaltu [3] = 0.25 * (1 + ksi) * (1 - eta)
                    funkcja_ksztaltu = np.array(funkcja_ksztaltu )
                    funkcja_ksztaltu_trans = np.resize(funkcja_ksztaltu , (4, 1))
                    N_mnozenie = funkcja_ksztaltu *funkcja_ksztaltu_trans
                    macierz = N_mnozenie * wagi[j]
                    sciana_prawa += macierz

            if i == 0:
                for j in range(2):
                    ksi = PC2[j][0]
                    eta = PC2[j][1]
                    funkcja_ksztaltu [0] = 0.25 * (1 + ksi) * (1 + eta)
                    funkcja_ksztaltu [1] = 0.25 * (1 - ksi) * (1 + eta)
                    funkcja_ksztaltu [2] = 0
                    funkcja_ksztaltu [3] = 0

                    funkcja_ksztaltu = np.array(funkcja_ksztaltu )
                    funkcja_ksztaltu_trans = np.resize(funkcja_ksztaltu, (4, 1))
                    N_mnozenie = funkcja_ksztaltu  * funkcja_ksztaltu_trans
                    macierz = N_mnozenie  * wagi[j]
                    sciana_gorna += macierz

            if i == 1:
                for j in range(2):
                    ksi = PC3[j][0]
                    eta = PC3[j][1]
                    funkcja_ksztaltu [0] = 0
                    funkcja_ksztaltu [1] = 0.25 * (1 - ksi) * (1 + eta)
                    funkcja_ksztaltu [2] = 0.25 * (1 - ksi) * (1 - eta)
                    funkcja_ksztaltu [3] = 0

                    funkcja_ksztaltu = np.array(funkcja_ksztaltu )
                    funkcja_ksztaltu_trans= np.resize(funkcja_ksztaltu , (4, 1))
                    N_mnozenie = funkcja_ksztaltu  * funkcja_ksztaltu_trans
                    macierz = N_mnozenie * wagi[j]
                    sciana_lewa += macierz

            if i == 2:
                for j in range(2):
                    ksi = PC4[j][0]
                    eta = PC4[j][1]
                    funkcja_ksztaltu[0] = 0
                    funkcja_ksztaltu[1] = 0
                    funkcja_ksztaltu[2] = 0.25 * (1 - ksi) * (1 - eta)
                    funkcja_ksztaltu[3] = 0.25 * (1 + ksi) * (1 - eta)

                    funkcja_ksztaltu = np.array(funkcja_ksztaltu)
                    funkcja_ksztaltu_trans = np.resize(funkcja_ksztaltu , (4, 1))
                    N_mnozenie = funkcja_ksztaltu  * funkcja_ksztaltu_trans
                    macierz = N_mnozenie  * wagi[j]
                    sciana_dolna += macierz
    macierzHBC += (sciana_prawa + sciana_gorna + sciana_lewa + sciana_dolna) * Alfaint * detJ
    return macierzHBC




def vectorP(element_arg):
    PC1 = [(1, -(sqrt(1 / 3))), (1, (sqrt(1 / 3)))]  # prawa
    PC2 = [((sqrt(1 / 3)), 1),  (-(sqrt(1 / 3)), 1)]  # gorna
    PC3 = [(-1, (sqrt(1 / 3))),  (-1, -(sqrt(1 / 3)))]  # lewa
    PC4 = [(-(sqrt(1 / 3)), -1), ((sqrt(1 / 3)), -1)]  # dolna
    wagi = [1,1]
    wymiar = (4,1)
    sciana_prawa = np.zeros(wymiar)
    sciana_gorna = np.zeros(wymiar)
    sciana_lewa = np.zeros(wymiar)
    sciana_dolna = np.zeros(wymiar)
    P = np.zeros(wymiar)
    funkcja_ksztaltu = [0 for x in range(4)]

    for i in range(-1, 3):
        x1 = node_list[element.Node_Id[i] - 1].x
        x2 = node_list[element.Node_Id[i + 1] - 1].x
        y1 = node_list[element.Node_Id[i] - 1].y
        y2 = node_list[element.Node_Id[i + 1] - 1].y
        detJ = sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2)) / 2

        if (element_arg.Node_Id[i] in BC) and (element_arg.Node_Id[i + 1] in BC):
            if i == -1:
                for j in range(2):
                    ksi = PC3[j][0]
                    eta = PC3[j][1]
                    funkcja_ksztaltu [0] = f1(ksi, eta)
                    funkcja_ksztaltu [1] = f2(ksi, eta)
                    funkcja_ksztaltu [2] = f3(ksi, eta)
                    funkcja_ksztaltu [3] = f4(ksi, eta)
                    funkcja_ksztaltu = np.array(funkcja_ksztaltu )
                    funkcja_ksztaltu_tran= np.resize(funkcja_ksztaltu , (4, 1))
                    funkcja_ksztaltu_tran*= Totint * wagi[j] * detJ
                    sciana_prawa += funkcja_ksztaltu_tran

            if i == 0:
                for j in range(2):
                    ksi = PC4[j][0]
                    eta = PC4[j][1]
                    funkcja_ksztaltu[0] = f1(ksi, eta)
                    funkcja_ksztaltu[1] = f2(ksi, eta)
                    funkcja_ksztaltu[2] = f3(ksi, eta)
                    funkcja_ksztaltu[3] = f4(ksi, eta)
                    funkcja_ksztaltu = np.array(funkcja_ksztaltu)
                    funkcja_ksztaltu_tran = np.resize(funkcja_ksztaltu, (4, 1))
                    funkcja_ksztaltu_tran *= Totint * wagi[j] * detJ
                    sciana_gorna += funkcja_ksztaltu_tran
            if i == 1:
                for j in range(2):
                    ksi = PC1[j][0]
                    eta = PC1[j][1]
                    funkcja_ksztaltu[0] = f1(ksi, eta)
                    funkcja_ksztaltu[1] = f2(ksi, eta)
                    funkcja_ksztaltu[2] = f3(ksi, eta)
                    funkcja_ksztaltu[3] = f4(ksi, eta)
                    funkcja_ksztaltu = np.array(funkcja_ksztaltu)
                    funkcja_ksztaltu_tran = np.resize(funkcja_ksztaltu, (4, 1))
                    funkcja_ksztaltu_tran *= Totint * wagi[j] * detJ
                    sciana_lewa += funkcja_ksztaltu_tran

            if i == 2:
                for j in range(2):
                    ksi = PC2[j][0]
                    eta = PC2[j][1]
                    funkcja_ksztaltu[0] = f1(ksi, eta)
                    funkcja_ksztaltu[1] = f2(ksi, eta)
                    funkcja_ksztaltu[2] = f3(ksi, eta)
                    funkcja_ksztaltu[3] = f4(ksi, eta)
                    funkcja_ksztaltu = np.array(funkcja_ksztaltu)
                    funkcja_ksztaltu_tran = np.resize(funkcja_ksztaltu, (4, 1))
                    funkcja_ksztaltu_tran *= Totint * wagi[j] * detJ
                    sciana_dolna += funkcja_ksztaltu_tran

    P += (sciana_prawa + sciana_gorna + sciana_lewa + sciana_dolna) * Alfaint

    return P


print(
    "-----------------------------------------------------------------------------------------------------------------")

#  wypisywanie
numer_iteracji = 1
for element in element_list:
    x0 = node_list[element.Node_Id[0] - 1].x
    x1 = node_list[element.Node_Id[1] - 1].x
    x2 = node_list[element.Node_Id[2] - 1].x
    x3 = node_list[element.Node_Id[3] - 1].x

    y0 = node_list[element.Node_Id[0] - 1].y
    y1 = node_list[element.Node_Id[1] - 1].y
    y2 = node_list[element.Node_Id[2] - 1].y
    y3 = node_list[element.Node_Id[3] - 1].y

    macierz_H = macierz(x0, x1, x2, x3, y0, y1, y2, y3)[0]
    element.H = macierz_H

    macierz_HBC = calculateHbc(element)
    element.HBC = macierz_HBC
    element.H_HBC += macierz_H + macierz_HBC



    wektor_p = vectorP(element)
    element.P = wektor_p

    macierz_C = macierz(x0, x1, x2, x3, y0, y1, y2, y3)[1]
    element.C = macierz_C

    print(f"Macierz H dla elementu ", numer_iteracji)
    macierz_H = np.array(macierz_H)
    print(macierz_H)
    print(f"Macierz C dla elementu",numer_iteracji)
    macierz_C = np.array(macierz_C)
    print(macierz_C)
    print("Macierz HBC dla elementu ", numer_iteracji)
    macierz_HBC = np.array(macierz_HBC)
    print(macierz_HBC)

    print("Wektor P dla elementu ",numer_iteracji)
    wektor_p = np.array(wektor_p)
    print(wektor_p)
    numer_iteracji += 1
print(
    "-----------------------------------------------------------------------------------------------------------------")
agregacja_h = agregateH()
print("Macierz H_zwykla agregacja:")
agregacja_h = np.array(agregacja_h)
print(agregacja_h)
print(
    "-----------------------------------------------------------------------------------------------------------------")
print("Macierz C agregacja:")
agregacja_c = agregateC()
agregacja_c = np.array(agregacja_c)
print(agregacja_c)
# for j in agregacjac:
#  print("\t".join(map(str, j)))
agregacja_P = agregateP()
agregacja_P = np.array(agregacja_P)
print("Wektor P agregacja")
print(agregacja_P)
agregacja_h_hbc = agregate_h_hbc()
agregacja_h_hbc = np.array(agregacja_h_hbc)
print("Agregacja H_finalna")
print(agregacja_h_hbc)

# symulacja
liczba_krokow = SimulationTimeint // SimulationStepTimeint

x = np.arange(Nodes_numberint,dtype=int).reshape(Nodes_numberint,1)
t0 = np.full_like(x,InitialTempint)
czas = SimulationStepTimeint
czas2 = czas
for i in range(liczba_krokow):
    macierz_H_uklad = agregacja_h_hbc + agregacja_c / SimulationStepTimeint
    macierz_H_uklad = np.array(macierz_H_uklad)
  #  print("macierzh/dt")
   # print(macierz_H_uklad)

    macierz_c_uklad = np.dot(agregacja_c / SimulationStepTimeint, t0)
    macierz_c_p_uklad = macierz_c_uklad + agregacja_P
    macierz_c_p_uklad = np.array(macierz_c_p_uklad)
    #print(";;;;;;;;;")
   # print(macierz_c_p_uklad)
    t1 = sp.linalg.solve(macierz_H_uklad, macierz_c_p_uklad)
    min = np.min(t1)
    max = np.max(t1)
    print("Time:",czas,"  Min:", min, "\t", "Max:", max)
    t0 = t1
    czas += czas2



