#!/usr/bin/env python
# coding: utf-8

path = input("Ingrese el la dirección del archivo") #"Try2020112923247120.csv"
delim = input("Delimitador de datos ")
dec = input("Delimitador de decimales ")
encode = input("Puede colorcar la codificación estandar ")
import pandas
 
if encode != "": data = pandas.read_csv(path, encoding=encode)
else: data = pandas.read_csv(path, delimiter=delim, decimal=dec)
print("Forma :",data.shape)

# ### Filtración de datos    
def get_columns(data):
    print(data.columns)
    global traits, individuals, species, abundance, treatment
    traits = input ("Nombre de las columnas de rasgos separados por comas").split(",") #["TotalHeight", "leaffreshweight_g" , "leafdryweight_g" , "Foliararea_cm2" , "wooddensity_core_g/cm3"] #
    r = False 
    while not r:
        for col in traits: r = (col in data.columns)
        if not r:  traits = input ("Hay columnas mal escritas o que no están en los datos\nNombre de las columnas de rasgos separados por comas").split(",") 
    
    individuals = input ("Nombre de la columna de individuos") #"ColectionNumber" #
    while individuals not in data.columns:  individuals = input ("\t¡Revisa!\nNombre de la columna de individuos")
    species = input ("Nombre de la columna de especies") #"species" #
    while species not in data.columns: species = input ("\t¡Revisa!\nNombre de la columna de especies") 
    abundance = input ("Nombre de la columna de abundancia \n(si no la hay deje en blanco)")#"individualnumberpersite" #
    
    while not (abundance in data.columns or abundance ==""): 
            print(abundance in data.columns or abundance =="")
            print(abundance)
            abundance = input ("\t¡Revisa!\nNombre de la columna de abundancia \n(si no la hay deje en blanco)")
    treatment = input ("Nombre de la columna para agrupar entre tratamientos") #BioRegion
    while not (treatment in data.columns or treatment ==""): treatment = input ("\t¡Revisa!\nNombre de la columna para agrupar entre tratamientos")
    
    #agregamos los valores de las columnas de interés a una sola columna que filtra los datos 
    columns = list(traits)
    columns.append(individuals)
    columns.append(species)
    if abundance != "": columns.append(abundance)
    if treatment != "": columns.append(treatment)
    return columns

#filtramos las columnas seleccionadas
Raw_matrix = data.filter(get_columns(data))
#Solo tomamos las filas sin datos faltantes
Raw_matrix = Raw_matrix.dropna()
#Convertimos los datos de rasgos a numeros operables (y también los datos de abundancia)
for i in Raw_matrix[traits].columns: Raw_matrix[i] = pandas.to_numeric(Raw_matrix[i])
if abundance != "": Raw_matrix[abundance] = pandas.to_numeric(Raw_matrix[abundance])
print(Raw_matrix.head)



#-------------------------------CALCULO DE LOS RAGOS--------------------------------#
#Esta función agrega una nueva columna a la matriz de rasgos x individuos según los datos dados (cocina los datos crudos)
#También agrega el rasgo a la lista de rasgos (para futuros cálculos)
rawcolumns = []
def cook_trait(T,n):
    global traits
    global rawcolumns
    print("\n¿ Calcular",T,"?")
    r = input("Sí (s) / No (n) ")
    while not (r=="s" or r=="n"): r = input("Sí (s) / No (n) ")
    if r=="s" and n in traits: print(T,"ya está agregado")
    elif r=="s" and n not in traits:
        print("Ingrese las columnas de la operación (puede agregar un factor de conversión separado de una coma)")
        #obtención del numerador
        a = input("Columna del numerador ").split(",")
        while a[0] not in traits: a = input("Columna del numerador ").split(",") #confirmación de datos, la columna dada debe estar dentro de las columnas de la matriz
        if len(a)==1: a.append(1) # si no se especifica un elemento de conversión, el factor de conversión será 1
        #obtención del denominador
        b = input("Columna del denominador ").split(",")
        while b[0] not in traits: b = input("Columna del denominador ").split(",") 
        if len(b)==1: b.append(1)
        Raw_matrix[n] = (Raw_matrix[a[0]]*int(a[1]))/(Raw_matrix[b[0]]*int(b[1]))

        traits.append(n) #añadimos el nuevo rasgo a la lista de rasgos
        if a[0] not in rawcolumns: rawcolumns.append(a[0]) #añadimos la columna que se usó como candidata a quitar
        if b[0] not in rawcolumns: rawcolumns.append(b[0])

            
#Dejamos a libertad del usuario quitar o no la columna (e.g. el area foliar aunque se usó en los cálculo debe quedarse por ser un rasgo funcional)
def quit_mesurments ():
    global Raw_matrix
    if len(rawcolumns)!=0: print("\n¡Rasgos calculados!")
    for i in rawcolumns:
        print("\n¿ Desea quitar",i,"de la matriz de datos ?")
        r = input("Sí (s) / No (n) ")
        while not (r=="s" or r=="n"): r = input("Sí (s) / No (n) ")
        if r=="s": 
            Raw_matrix = Raw_matrix.drop(columns=[i]) #Se elemina de la matriz
            traits.remove(i) #también se elimina de la lista de rasgos
        elif r=="n" : pass
#llamamos la función que calcula los rasgos
print("\n\n\t\tCALCULO DE RASGOS FUNCIONALES\n")
cook_trait("Area Foliar específica","SLA")
cook_trait("Contenido foliar de materia seca","LDMC")
cook_trait("Densidad de madera","WD")
cook_trait("Proporcines de nutrientes foliares N:P","N:P")
quit_mesurments()


#----------------------------Obtención de matriz de rasgos específicos------------------------------##

#primer promedio entre hojas para cada individuos
if treatment != "": Trait_matrix = Raw_matrix.groupby([treatment,species,individuals]).mean() 
else: Trait_matrix = Raw_matrix.groupby([species,individuals])

#Es un dato que no debia promediarse
if abundance != "": Trait_matrix = Trait_matrix.drop(columns=[abundance]) 

#Luego se promedia entre individuos de cada especie (manteniendo la agrupación por tratamiento)
if treatment != "": Trait_matrix = Trait_matrix.groupby([treatment,species]).mean() 
else: Trait_matrix = Trait_matrix.groupby([species]).mean()

if abundance != "":
    if treatment != "": Abun_matrix = Raw_matrix.groupby([treatment,species]).sum()
    else:  Abun_matrix = Raw_matrix.groupby([species]).sum()

Abun_matrix = Abun_matrix.drop(columns=traits)
Abun_matrix = Abun_matrix.reset_index()
Abun_matrix = Abun_matrix.set_index(species)


(Trait_matrix.join(Abun_matrix[abundance],on =[species])).to_csv("out_tables\Trait_matrix.csv")
print("\n\t\tSE HA GENERADO UNA MATRIZ DE RASGOS")

#Estandarización de los datos
for column in Trait_matrix:   
    Trait_matrix[column] = (Trait_matrix[column]-Trait_matrix[column].mean())/Trait_matrix[column].std()
#import sklearn
#sklearn.StandardScaler(Trait_matrix)
(Trait_matrix.join(Abun_matrix[abundance],on =[species])).to_csv("out_tables\Transformed-Trait_matrix.csv")
print("\n\t\tSE HA GENERADO UNA MATRIZ DE RASGOS TRANSFORMADOS")
#---------------------------------PRUEBAS DE NORMALIDAD----------------------------------#
from scipy.stats import kstest
print("\t\tPRUEBAS DE NORMALIDAD")
for group, column in Trait_matrix[traits].groupby([treatment]):
    print("\tPruebas de normalidad para",group)
    r = ""
    for t in column.columns: 
        ks = kstest(column[t],"norm")[1]
        if ks<0.01:
            r+=t+" NO es normal "+str(ks)+" \n"
        elif ks>0.01 : 
            r+=t+" es normal\n"
    print("-------------------------------\n")
    if not "no " in r: print("¡Todo parece ser normal!\n")
    else: print(r)
    print("--------------------------------\n")


#------------------- Cálculo de Matriz de componentes principales -----------------------------#
import scipy.linalg as la
from math import log

CV = Trait_matrix.cov()

vals,vects = la.eig(CV)
perct = [] #Porcentaje de varianza explicada
for i in vals: 
    perct.append((i*100)/vals.sum())

p=len(traits) #número de variames
N=Trait_matrix[[traits[0]]].size #Tamaño muestral
def eme (m,lg=False): #la sumatoria de los valores propios hasta m
    s=0
    if lg: #si es la suma de los logaritmos
        for i in range(m,len(vals)+1): s+=log(vals[i-1].real,10)
    else: 
        for i in range(m,len(vals)+1): s+=vals[i-1].real
    return s
x_2 = 0 
gl = 0
for m in range(len(vals)):
    x_2 = (((p-m)*(N-1)*log((eme(m+1))/(p-m),10))-((N-1)*(eme(m+1,lg=True))))
    gl = (0.5*((p-m)*(p-m+1))-1)
    print("El componente número",m+1,"explica una varianza del",round(perct[m].real),"%\nTiene un X2 (análisis de Anderson) de",x_2,"con grados de libertad",gl,"\n")

desicion = input("Revise en la tabla de Chi 2 y defina hasta qué componente es significativa una varianza explicada ")
PC = vects[:,:int(desicion)]
PC = pandas.DataFrame(PC,index=traits)
PC.to_csv("out_tables\Traits_loading.csv")
print("\n\n\t\tCOMPONENTES PRINCIPALES\n\t\t (loading de las variables)\n",PC)
## Obtención del gráfico del PCA

#-------------OBTENCIÓN DE LA NUEVA MATRIZ DE ESPECIES EN LOS DOS COMPONENTES--------#
#se convierte la matriz agrupada a una matrix normal (pandas dataframe)
M = Trait_matrix.reset_index()
def cpdata(cp, Variables):
    Axis = []
    for index, rows in M.iterrows(): 
        Axis . append(rows[Variables].dot(PC[cp-1]))
    #Axis = pandas.Series(Axis).values
    return Axis

T= pandas.DataFrame({"PC1":cpdata( Variables = traits, cp = 1),"PC2":cpdata( Variables = traits, cp = 2)})

T = T.join(M[[treatment,species]])
#Datos atípicos
#T = T.drop(T[(T[treatment] == "Valle del Cauca") & (T[species] == "Tabebuia rosea")].index)
#T = T.drop(T[(T[treatment] == "Caribe") & (T[species] == "Swietenia macrophylla")].index)

#Se establecen los indices
T = T.set_index(species)
T.to_csv("out_tables\species_scores.csv")
print("\n\n\t\tScores de las espcies en los componentes 1 y 2\n\n",T)

#----------------------------------GRAFICO DE LOS VECTORES PROPIOS Y DE LAS ESPECIES----------------------------------#
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import seaborn as sn
sn.set()

def Trait_plot():

    plt.xlabel("PC{} ".format(1)+str(round(perct[0].real))+" %") #Nombre del eje x
    plt.ylabel("PC{} ".format(2)+str(round(perct[1].real))+" %") #Nombre del eje y
    if treatment != "":
        for i,j in T.groupby(treatment): plt.plot(j["PC1"] ,j["PC2"],'o',markersize=8)
        
        #leyenda por tratamientos
        global treatments
        treatments = M[treatment].unique()
        plt.legend(treatments) 

    else: plt.plot(T["PC1"] ,T["PC2"],'o',markersize=8)
            
    global fig
    
    fig = plt.gcf()
    fig.set_size_inches(10.5, 7.5)
    return fig, plt.gca().lines[-1].get_color()
    
#se grafican los vectores propios
def d_vectors():
    for i in range(vects.shape[0]):
        plt.arrow(0, 0, vects[i,0]*3, vects[i,1]*3,color = 'brown')
        plt.text(vects[i,0]* 3.2, vects[i,1] * 3.2, traits[i], color = 'black', ha = 'center', va = 'center')

sn.set(font_scale=1.25)
sn.set_style("white")
fig,cor=Trait_plot()
d_vectors()
#Se guardan las graficas que se obtienen
fig.savefig('out_graphics\PCA.png', dpi=100)
print("\n\t\tSe ha generao un grafico del PCA")
fig.show()

#----------------------------------ANALISI DE DIVERSIAD----------------------------------#
from scipy.spatial import ConvexHull
from matplotlib.collections import PolyCollection
import numpy as np

def FRichness(points,p=None): #Está función determina el ConvexHull de un conjunto de puntos
    
    #Se obtienen las dos columnas de interés y se convierten a una matriz numérica (numpy)
    points = points[["PC1","PC2"]]
    points =  points.values 
    
    #La función ConvexHull halla el ConvexHull
    hull = ConvexHull(points)  
    
    #Una etiqueta para cada tratamiento
    if p == None:  
        labeli="Total"
        cor = "gray"
    else: labeli = treatments[p]
    
    #Se obtienen solo los vertices del CovexHull
    circular_hull_verts = np.append(hull.vertices,hull.vertices[0])
    
    #Se grafica cada tratamiento (con una etiqueta para luego colocar en la leyenda)
    if not p==None:   
        plt.plot(points[:,0] ,points[:,1],'o',markersize=8, label =labeli) 
        cor = plt.gca().lines[-1].get_color()
        
    
    #Se grafican las lineas entre esos vertices
    plt.plot(points[:,0][circular_hull_verts], points[:,1][circular_hull_verts], lw=2, zorder=-1,
                    c=cor) #para que tengan el mismo color

    #Se rellena este ConvexHull
    plt.fill(points[hull.vertices,0], points[hull.vertices,1], alpha=0.05,
                 c=cor) #Para que tengan el mismo como
    
    return hull.area #La Riqueza funcional de la comunidad (tratamiento) será el area del Covex Hull Calculado
plt.clf()
#Propiedades del gráfico
#-----------------------
#Tamaño

figu = plt.gcf()
figu.set_size_inches(10.5, 7.5)
#Ejes
plt.xlabel("PC{} ".format(1)+str(round(perct[0].real))+" %") #Nombre del eje x
plt.ylabel("PC{} ".format(2)+str(round(perct[1].real))+" %") #Nombre del eje y
plt.title("Convex Hull")

R=[]
#Se llama la función de para cada tratamiento
if treatment != "":
    for t in range(len(treatments)):  R.append(FRichness(T[T[treatment] == treatments[t]],t))
    #leyenda del gráfico
    plt.legend()
else: pass




#Se guardan las graficas que se obtienen
sn.set_style("white")
sn.set(font_scale=1.25)
figu.savefig('out_graphics\PCA_ConvexHull.png', dpi=100)

#Se almacena el índice calculado
Div = pandas.DataFrame({treatment:treatments,"FRichness":R})
Div.loc[t+1]=["Total"]+[FRichness(T)]
Div.to_csv("out_tables\Div_indices.csv")
print("\n\n\t\tÍNDICES DE DIVERSIDAD FUNCIONAL\n",Div)


#Análisis de probabilidad de Kernel basado en la abundancia

if abundance != "":
    import seaborn as sn
    sn.set()

    TPD = T.join(Abun_matrix[abundance])

    fig, axes = plt.subplots(1, 2,figsize=(20,10))
    sn.set(font_scale=2)
    axes[0].set_title("Densidad de rasgos sin la abundancia")
    axes[1].set_title("Densidad de rasgos Con la abundancia")

    sn.kdeplot(TPD["PC1"],TPD["PC2"],ax = axes[1],weights=TPD[abundance]
               ,shade=True,n_levels=100, color ="w")
    sn.kdeplot(TPD["PC1"],TPD["PC2"],ax = axes[0]
               ,shade=True,n_levels=100, color ="w")
    d_vectors()
    fig.savefig('out_graphics\Trait_Probability_Densitiy.png', dpi=100)
    #fig.show()

