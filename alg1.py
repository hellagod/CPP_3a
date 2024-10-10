def greedy_naive_solution(weights, n):
    cliques = []
    edges = [(i, j, weights[i][j]) for i in range(n) for j in range(i + 1, n)]
    edges.sort(key=lambda x: x[-1], reverse=True)
    while edges:
        new_clique = []
        edge = edges.pop(0)
        if edge[2] < 0:
            break
        new_clique.append(edge)
        new_edges = list(filter(lambda e: not (e[0] in edge or e[1] in edge),edges))
        cliques.append(new_clique)
        edges = new_edges
    sol = {}
    for k, clique in enumerate(cliques):
        sol[k] = set()
        for i, j, w in clique:
            sol[k].add(i)
            sol[k].add(j)
    for i, v in enumerate(set(i for i in range(n)) - set().union(*sol.values())):
        sol[i + len(cliques)] = [v]
    for k, clique in enumerate(cliques):
        sol[k] = list(sol[k])
    return sol


test_n = 6
test_w = [[-10, -3, -10, 4, -6, -1],
          [10, -9, -10, 3, -4, 4],
          [8, 0, -3, -2, -2, 4],
          [0, -9, 9, 7, -5, -4],
          [-5, 4, -5, 2, 8, 1],
          [6, -1, 10, 10, 3, -3]]

init_solution = greedy_naive_solution(test_w, test_n)

print(init_solution)
