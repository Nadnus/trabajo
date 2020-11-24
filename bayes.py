import pandas as pd
import itertools
import math
import random
import copy

# marginal


def count(df, values, recurrences):
    dfValues = df.values
    dfColumns = df.columns
    res = []
    index = -1
    for i in range(len(values)):
        count = 0
        for j in range(len(dfColumns)):
            if dfColumns[j] == values[i]:
                index = j
        for k in range(len(dfValues)):
            if(dfValues[k][index] == recurrences[i]):
                count += 1
        res.append(count)
    return res


def createQueue(df, X):
    index = -1
    queue = []
    for i in range(len(df.columns)):
        if df.columns[i] == X:
            index = i
    for j in range(len(df.values)):
        if df.values[j][index] not in queue:
            queue.append(df.values[j][index])
    return queue


def estimar_marginal(df, X, alpha):
    queue = createQueue(df, X)
    size = len(df.values)
    res = []
    for i in queue:
        freqX = count(df, [X], [i])
        f = (freqX[0] + alpha)/(size+(alpha*len(queue)))
        res.append(f)

    result = {}
    for j in range(len(queue)):
        # insideList = []
        # insideList.append(queue[j])
        # insideList.append(res[j])
        result[queue[j]] = res[j]

    return result


def conteo_interseccion(df, parametros, columnas):
    col1 = df[columnas[0]]
    col2 = df[columnas[1]]
    cont = 0
    for i in range(len(col1)):
        if(col1[i] == parametros[0] and col2[i] == parametros[1]):
            cont += 1
    return cont


def cardinalidad(df, columnaX):
    cardinality = len(pd.Index(df[columnaX]).value_counts())
    return cardinality


def conteo(df, x, columnaX):
    cont = 0
    for i in df[columnaX]:
        if i == x:
            cont += 1

    return cont


def findVariables(df, columnaX):
    l = []
    for elem in df[columnaX]:
        if elem not in l:
            l.append(elem)
    return l


def probabilidad_x_si_y(df, x, columnaY, valY, a):
    toReturn = []
    columnas = [x, columnaY]
    # faltan separar las variables en variables []
    variables = findVariables(df, x)
    for row in variables:
        parametros = [row, valY]
        card = conteo_interseccion(df, parametros, columnas) + a
        denom = df.size + cardinalidad(df, x)*cardinalidad(df, columnaY)*a
        numerador = card / denom
        # confirmar los valores del segundo denominador, en la pizarra sale c * y * a,  sera cy *a?
        denom2 = (conteo(df, valY, columnaY) + a) / \
            (df.size + cardinalidad(columnaY) * a)
        totes = numerador / denom2
        toReturn.append(totes)
    return toReturn


def intersect_count(df, valores, columnas):
    l = len(valores)
    if l == 1:
        return conteo(df, valores[0], columnas[0])
    rows = []
    for tup in df.itertuples():
        var1 = getattr(tup, columnas[0])
        var2 = getattr(tup, columnas[1])
        e1 = var1 == valores[0]
        e2 = var2 == valores[1]
        if e1 and e2:
            rows.append(tup)
    for i in range(2, l):
        to_remove = []
        for tup in rows:
            a = getattr(tup, columnas[i])
            if a != valores[i]:
                to_remove.append(tup)
        for tup in to_remove:
            rows.remove(tup)
    return len(rows)


def condicional(df, target_var, variables, a, flag=False):
    if not variables:
        return estimar_marginal(df, target_var, a)
    df_len = len(df)
    valores = {}
    valores[target_var] = findVariables(df, target_var)
    for var in variables:
        buffer = findVariables(df, var)
        valores[var] = buffer
    lista_vars = []
    for v in valores:
        lista_vars.append(valores[v])
    lista_trunca = lista_vars[1:]
    combos = list(itertools.product(*lista_vars))
    otros_combos = list(itertools.product(*lista_trunca))

    resultados = {}
    card_acum = a
    for elem in variables:
        card_acum = card_acum * cardinalidad(df, elem)
    card_acum_x = card_acum * cardinalidad(df, target_var)
    for combo in combos:
        combo2 = combo[1:]
        numerador1 = intersect_count(df, combo, [target_var]+variables) + a
        denom1 = df_len + card_acum_x
        num2 = intersect_count(df, combo2, variables) + a
        denom2 = df_len + card_acum
        frac1 = (numerador1/denom1)
        frac2 = (num2/denom2)
        resultados[combo] = frac1/frac2
    for c in otros_combos:
        acc = 0
        for val in valores[target_var]:
            key = (val,) + c
            acc = acc + resultados[key]
        for val in valores[target_var]:
            resultados[(val,) + c] = resultados[(val,) + c]/acc
    if flag:
        return ([[target_var, variables], resultados])

    return ((resultados))


def condicionalFijas(df, fijas_nombre, fijas, variables_usadas=[]):
    valid_combos = []
    if not variables_usadas:
        variables_usadas = df.columns
    valid_values = []
    for i in range(len(variables_usadas)):
        to_add = []
        if variables_usadas[i] not in fijas_nombre:
            to_add.append(findVariables(df, variables_usadas[i]))

        else:
            index = fijas_nombre.index(variables_usadas[i])
            to_add.append([fijas[index]])
        valid_values.append(to_add[0])
    valid_combos = list(itertools.product(*valid_values))
    return valid_combos


def isDAG(g):  # g es una matriz de adyacencia
    visited = []
    q = [0]
    while len(q) != 0:
        curr = q[0]
        for i in range(len(g)):
            if g[i][curr] == 1:
                if i in visited:
                    return False
                else:
                    q.append(i)
        visited.append(curr)
        q.pop(0)
    return True


def addEdge(g):
    l = []
    for i in range(len(g)):
        for j in range(len(g)):
            if i != j and g[i][j] == 0:
                aux = copy.deepcopy(g)
                aux[i][j] = 1
                if isDAG(aux):
                    l.append(aux)
    return l


def rmvEdge(g):
    l = []
    for i in range(len(g)):
        for j in range(len(g)):
            if g[i][j] == 1:
                aux = copy.deepcopy(g)
                aux[i][j] = 0
                l.append(aux)
    return l


def invEdge(g):
    l = []
    for i in range(len(g)):
        for j in range(len(g)):
            if g[i][j] == 1:
                aux = copy.deepcopy(g)
                aux[i][j] = 0
                aux[j][i] = 1
                if isDAG(aux):
                    l.append(aux)
    return l


def getParents(varName, df, g):
    colNames = list(df.columns)
    idx = colNames.index(varName)
    result = [varName]
    aux = []
    for i in range(len(g)):
        if(g[idx][i] == 1):
            aux.append(colNames[i])
    result.append(aux)
    return result


def calcQ(varName, df, g):
    prod = 1
    parents = getParents(varName, df, g)[1]
    if len(parents) > 0:
        for p in parents:
            prod *= cardinalidad(df, p)
    return prod


def Qpossibilities(List):
    res = []
    for r in itertools.product(*List):
        aux = []
        for i in range(len(r)):
            aux.append(r[i])
        res.append(aux)
    return res


def Nij(df, i, j, g):
    colNames = list(df.columns)
    parent = getParents(colNames[i], df, g)
    result = 0

    L = []

    for p in parent[1]:
        L.append(findVariables(df, p))

    possibilities = Qpossibilities(L)
    x = possibilities[j]
    for index, row in df.iterrows():
        a = []
        for i in range(len(parent[1])):
            if(row[colNames[i]] == x[i]):
                a.append(True)
            else:
                a.append(False)

        if(all(a)):
            result += 1
    return result


def Nijk(df, i, j, k, g):
    colNames = list(df.columns)
    parent = getParents(colNames[i], df, g)
    result = 0
    iPossibilities = findVariables(df, colNames[i])

    L = []

    for p in parent[1]:
        L.append(findVariables(df, p))

    possibilities = Qpossibilities(L)
    x = possibilities[j]

    for index, row in df.iterrows():
        a = []
        for i in range(len(parent[1])):
            if(row[colNames[i]] == x[i]):
                a.append(True)
            else:
                a.append(False)
        if(all(a) and row[colNames[i]] == iPossibilities[k]):
            result += 1
    return result


def conteo(df, x, columnaX):
    cont = 0
    for i in df[columnaX]:
        if i == x:
            cont += 1
    return cont


def findVariables(df, columnaX):
    l = []
    for elem in df[columnaX]:
        if elem not in l:
            l.append(elem)
    return l


def cardinalidad(df, columnaX):
    cardinality = len(pd.Index(df[columnaX]).value_counts())
    return cardinality


def intersect_count(df, valores, columnas):
    l = len(valores)
    if l == 1:
        return conteo(df, valores[0], columnas[0])
    rows = []
    for tup in df.itertuples():
        var1 = getattr(tup, columnas[0])
        var2 = getattr(tup, columnas[1])
        e1 = var1 == valores[0]
        e2 = var2 == valores[1]
        if e1 and e2:
            rows.append(tup)
    for i in range(2, l):
        to_remove = []
        for tup in rows:
            a = getattr(tup, columnas[i])
            if a != valores[i]:
                to_remove.append(tup)
        for tup in to_remove:
            rows.remove(tup)
    return len(rows)


def condicional(df, target_var, variables, a, flag=False):
    if not variables:
        return estimar_marginal(df, target_var, a)
    df_len = len(df)
    valores = {}
    valores[target_var] = findVariables(df, target_var)
    for var in variables:
        buffer = findVariables(df, var)
        valores[var] = buffer
    lista_vars = []
    for v in valores:
        lista_vars.append(valores[v])
    lista_trunca = lista_vars[1:]
    combos = list(itertools.product(*lista_vars))
    otros_combos = list(itertools.product(*lista_trunca))

    resultados = {}
    card_acum = a
    for elem in variables:
        card_acum = card_acum * cardinalidad(df, elem)
    card_acum_x = card_acum * cardinalidad(df, target_var)
    for combo in combos:
        combo2 = combo[1:]
        numerador1 = intersect_count(df, combo, [target_var]+variables) + a
        denom1 = df_len + card_acum_x
        num2 = intersect_count(df, combo2, variables) + a
        denom2 = df_len + card_acum
        frac1 = (numerador1/denom1)
        frac2 = (num2/denom2)
        resultados[combo] = frac1/frac2
    for c in otros_combos:
        acc = 0
        for val in valores[target_var]:
            key = (val,) + c
            acc = acc + resultados[key]
        for val in valores[target_var]:
            resultados[(val,) + c] = resultados[(val,) + c]/acc
    if flag:
        return ([[target_var, variables], resultados])

    return ((resultados))

# entropia


def generate_random_m(size):
    mat = [[random.randint(0, 1) for _ in range(size)]for _ in range(size)]
    return mat


def find_aristas(labels, matrix):
    indice_x = 0
    to_return = []
    for variable_origen in matrix:
        indice_j = 0
        for variable_target in variable_origen:
            if variable_target == 1:
                to_return.append((labels[indice_x], labels[indice_j]))
            indice_j = indice_j + 1
        indice_x = indice_x + 1

    return to_return


def entropia(df, matrix, alpha):
    sumatoria = 0

    variables = {}
    for col in df.columns.values:
        variables[col] = findVariables(df, col)
    for vertice in df.columns.values:
        marginales = estimar_marginal(df, vertice, alpha)
        for resultado in marginales:

            val = (math.log(marginales[resultado])) * \
                (conteo(df, resultado, vertice))
            sumatoria = sumatoria + val

    aristas = find_aristas(df.columns.values, matrix)
    for a in aristas:
        origen = a[0]
        final = a[1]
        if origen == final:
            continue
        # llenar la formula de el log
        condicional_combos = condicional(df, origen, [final], alpha)
        lista = [variables[origen], variables[final]]
        combos = list(itertools.product(*lista))
        for c in combos:
            cond = condicional_combos[c]
            a = (math.log(cond))
            b = (conteo_interseccion(df, c, df.columns.values))
            val = a*b
            sumatoria = sumatoria + val
    return sumatoria


def AIC_k(df, g):
    size = len(g)
    colNames = list(df.columns)
    accum = 0
    for i in range(size):
        Q = calcQ(colNames[i], df, g)
        accum += ((cardinalidad(df, colNames[i]) - 1) * Q)
    return accum


def AIC(df, g, alpha):
    return entropia(df, g, alpha)+AIC_k(df, g, alpha)


def MDL(df, g, alpha):
    N = len(df.index)
    return entropia(df, g, alpha) + (AIC_k(None, None, alpha))/2*math.log(N, 2)


def k2(df, g):
    colNames = list(df.columns)
    accum1 = 1

    for i in range(len(colNames)):
        accum2 = 1
        for j in range(calcQ(colNames[i], df, g)):
            num = math.factorial(cardinalidad(
                df, colNames[i])-1 + Nij(df, i, j, g))
            denom = math.factorial(cardinalidad(
                df, colNames[i])-1 + Nij(df, i, j, g))

            accum3 = 1
            for k in range(cardinalidad(df, colNames[i])):
                accum3 *= math.factorial(Nijk(df, i, j, k, g))

            accum2 *= (num/denom)*accum3
        accum1 *= accum2
    return accum1


def hillClimbing(df, initialState):  # initialState tiene la misma forma que g
    state = initialState
    auxState = state
    currScore = k2(df, state)
    auxScore = currScore

    while(1):
        neighbors = addEdge(state)+rmvEdge(state)+invEdge(state)
        for n in neighbors:
            auxScore = k2(df, n)
            print("curr", currScore)
            print("aux", auxScore)
            if auxScore > currScore:
                auxState = copy.deepcopy(n)
                currScore = auxScore
        if state == auxState:
            return state
        else:
            state = copy.deepcopy(auxState)
