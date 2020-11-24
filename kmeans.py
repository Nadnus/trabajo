import math
import collections
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import scatter
import pandas as pd
# observaciones tiene la forma de ((x1,y1,z1),(x2,y2,z2))

# findCentroid es una funcion que encuentra el hipotetico centroide de un set de observaciones, encontrando su promedio


def findCentroid(observaciones):
    n_dimensiones = len(observaciones[0])
    centroide = []
    for dimension in range(n_dimensiones):
        centroide.append(0)
        for obs in observaciones:
            current = obs[dimension]
            centroide[dimension] = centroide[dimension] + current

        centroide[dimension] = centroide[dimension]/len(observaciones)
    return centroide


def euclidDistance(coord_list_a, coord_list_b):
    accum = 0
    dimensions = len(coord_list_a)
    for dimension in range(dimensions):
        line = (coord_list_a[dimension] - coord_list_b[dimension]) * \
            (coord_list_a[dimension] - coord_list_b[dimension])
        accum += line
    root = math.sqrt(accum)
    return root

# findNearestMean itera sobre las medias y retorna la que este mas cercana por distancia eculideana


def findNearestMean(observacion, means):
    min_distance = 999999
    n_mean = []
    for mean in means:
        distancia = euclidDistance(observacion, mean)
        if distancia < min_distance:
            n_mean = mean
            min_distance = distancia
    return n_mean

# se retorna un diccionario con las medias como llaves, y la lista de observaciones que la tienen como mas cercana como value


def findAllNearestMeans(observaciones, means):
    observaciones_per_mean = collections.defaultdict(list)
    for observacion in observaciones:
        key = findNearestMean(observacion, means)
        buffer = observaciones_per_mean[tuple(key)]
        buffer.append(observacion)
    return observaciones_per_mean

# RandomK es un procedimiento simple que elije un sample aleatorio de los que se tienen; y los inicializa como means.


def generateRandomK(k, observaciones):
    medias = random.sample(observaciones, k)
    return medias


def plotMeans2d(dict_medias):
    dimensiones = [[], [], []]
    media_colors = {}
    unique_medias = list(set(dict_medias.keys()))
    step_size = (256**3)//len(unique_medias)
    for i, m in enumerate(unique_medias):
        media_colors[m] = '#{}'.format(hex(step_size*i)[2:])
    for media in dict_medias:
        for observacion in dict_medias[media]:
            dimensiones[2].append(tuple(media))
            for dimension in range(len(observacion)):
                dimensiones[dimension].append(observacion[dimension])
    print('el pepe')
    print(dimensiones[2])
    colors = [media_colors[current_media] for current_media in dimensiones[2]]
    for item in range(len(colors)):
        if colors[item] == '#0':
            colors[item] = '#000000'
    #tr = plt.scatter(dimensiones[0], dimensiones[1],  alpha=0.5)
    tr = plt.scatter(dimensiones[0], dimensiones[1], c=colors, alpha=0.5)
    plt.show()


def kmeans(k, observaciones):
    # encontramos la configuracion inicial
    medias = generateRandomK(k, observaciones)
    cercanas = findAllNearestMeans(observaciones, medias)
    # se sigue iterando hasta que se deje de cambiar la media despues de una iteracion
    while True:
        # cada loop se usa la lista de centroides para ver las medias updateadas
        centroides = []
        for media in medias:
            # para cada cluster, se encuentra su centroide
            c = cercanas[tuple(media)]
            centroides.append(findCentroid(c))
        if medias == centroides:
            break
        medias = centroides
        cercanas = findAllNearestMeans(observaciones, medias)
    return medias


def dfToList(df):
    lista = []
    for row in df.itertuples():
        lista.append(list(row))
    return lista

df = pd.read_csv('incomeProcessed.csv')
mapa = dfToList(df)
result = kmeans(4, mapa)
print(result)
#myplot = plotMeans2d(result)
