import numpy as np


def get_edge(i, j):
    return [i, j] if i < j else [j, i]


def get_weights(w, n, v):
    if v == 0:
        return np.concatenate(([0], w[v, 1:]))
    left = w[:v, v]
    right = w[v, v + 1:] if v < n - 1 else []
    return np.concatenate((left, [0], right))


def find_clique(cliques, i):
    for idx, clique in cliques.items():
        if i in clique:
            return idx


def get_delta(w, cliques, cur_clique, new_clique, i):
    delta = 0
    for v in cliques[new_clique]:
        edge = get_edge(i, v)
        delta += w[*edge]
    for v in cliques[cur_clique]:
        if v != i:
            edge = get_edge(i, v)
            delta -= w[*edge]
    return delta


def get_delta_dummy_clique(w, cliques, cur_clique, i):
    delta = 0
    for v in cliques[cur_clique]:
        if v != i:
            edge = get_edge(i, v)
            delta -= w[*edge]
    return delta


def initialize_matrix(w, cliques):
    n = len(w)
    num_cliques = len(cliques)
    m = np.zeros((n, num_cliques + 1))

    for i in range(n):
        my_clique = find_clique(cliques, i)
        for j in range(num_cliques):
            if my_clique != j:
                m[i, j] = get_delta(w, cliques, my_clique, j, i)
        # отдельно, т.к. фиктивных клик нет в словаре cliques
        m[i, num_cliques] = get_delta_dummy_clique(w, cliques, my_clique, i)

    return m


def obj_function(w, cliques):
    total_weight = 0
    for clique in cliques.values():
        for i in clique:
            for j in clique:
                if i != j and i < j:
                    total_weight += w[i][j]
    return total_weight

def modify_matrix(m, w, cliques, n, num_cliques, i, j, k):
    # i - индекс вершины, j - индекс клики "откуда", k - индекс клики "куда"
    cur_clique, new_clique = cliques[j], cliques[k]

    m[i] = [x - m[i, k] for x in m[i]]

    # изменение приращений клик "откуда" и "куда"
    for v in range(n):
        if v != i:
            edge = get_edge(i, v)
            m[v, j] -= w[*edge]
            if k != num_cliques:
                m[v, k] += w[*edge]

    # подшивание столбца
    if k == num_cliques:
        new_column = get_weights(w, n, i)
        for v in range(n):
            if v != i:
                v_weights = get_weights(w, n, v)
                v_weights[i] = 0
                new_column[v] -= sum(v_weights)
        np.insert(m, num_cliques, new_column, axis=1)

    # убирание столбца
    if len(cur_clique) == 1:
        m = np.delete(m, j, axis=1)

    # пересчет строк соседей
    for v in cur_clique:
        if v != i:
            edge = get_edge(i, v)
            m[v] = [x + w[*edge] for x in m[v]]

    for v in new_clique:
        edge = get_edge(i, v)
        m[v] = [x - w[*edge] for x in m[v]]

    return m