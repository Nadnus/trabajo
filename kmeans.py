import math
import collections
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import scatter
import pandas as pd
from sklearn import preprocessing
import numpy as np

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


def plotMeans2d(dict_medias, figura, c1, c2):
    c1 = int(c1)
    c2 = int(c2)
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

    print(dimensiones[2])
    colors = [media_colors[current_media] for current_media in dimensiones[2]]
    for item in range(len(colors)):
        if colors[item] == '#0':
            colors[item] = '#000000'
    #tr = plt.scatter(dimensiones[0], dimensiones[1],  alpha=0.5)
    figura[c1][c2].scatter(dimensiones[0], dimensiones[1], c=colors, alpha=0.5)


# le paso basicamente uno de los dataframes post split


def noisify(x):
    if type(x) == str:
        return x
    else:
        return x + np.random.randn()*0.1


def reduce2d(dimension1, dimension2, dict_medias):
    nuevo_dict = {}
    for media in dict_medias:
        d1 = media[dimension1]
        d2 = media[dimension2]
        nueva_media = tuple([d1, d2])
        nuevo_dict[nueva_media] = []
        for observacion in dict_medias[media]:

            observacion_reducida = [
                noisify(observacion[dimension1]), noisify(observacion[dimension2])]
            nuevo_dict[nueva_media].append(observacion_reducida)
    return nuevo_dict


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
    return cercanas


def dfToList(df):
    lista = []
    for row in df.itertuples():
        lista.append(list(row[1:]))
    return lista


def centroidFreq(d):
    freq_dict = {}
    for centroid in d:
        freq_dict[centroid] = len(d[centroid])
    return freq_dict


df = pd.read_csv('incomeProcessed.csv', index_col=False)
original_columns = df.columns
print((df.columns)[0])
x = df.values  # returns a numpy array
# Normalizamos la data
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)

educaciones = set(df[2].to_list())
df = df.sample(3000)
grouped = df.groupby(df[2])
frames = {}
for val in educaciones:

    frames[val] = (grouped.get_group(val))


resultados = {}
dimensiones = [5, 7]
fig, figura = plt.subplots(2, math.ceil(len(frames)/2))
i = 0
for f in sorted(frames):
    y = int(i/2)
    x = i % 2
    mapa = dfToList(frames[f])
    result = kmeans(3, mapa)
    resultados[f] = (result)
    # Escojer las dimensiones a comparar
    noisy_frame = reduce2d(dimensiones[0], dimensiones[1], result)
    plotMeans2d(noisy_frame, figura, x, y)
    figura[x][y].set_title(i)
    figura[x][y].set(xlabel=original_columns[dimensiones[0]],
                     ylabel=original_columns[dimensiones[1]])
    i += 1
plt.show()
