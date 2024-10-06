def greedy_naive_solution(n, weights):
    cliques = []
    edges = [(i, j, weights[i][j]) for i in range(n) for j in range(i + 1, n)]
    edges.sort(key=lambda x: x[-1], reverse=True)
    used_edges = []
    while edges:
        new_clique = []
        edge = edges.pop(0)
        new_clique.append(edge)
        for e in edges[:]:
            if not any((e[0] in pair or e[1] in pair) for pair in used_edges):
                new_clique.append(e)
                used_edges.append(e)
                break
            edges.remove(e)
        cliques.append(new_clique)
    naive_sol = {}
    for clique in cliques:
        for i, j, w in clique:
            naive_sol[(i, j)] = 1
    return naive_sol
