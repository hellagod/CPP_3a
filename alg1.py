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


test_n = 10
test_w =[[0.273, 0.433, 0.665, 0.595, 0.957, 0.381, 0.756, 0.823, 0.498, -0.164],
           [0.433, -0.095, -0.44, 0.97, 0.751, 0.7, -0.179, -0.161, 0.438, -0.886],
           [0.665, -0.44, 0.826, -0.191, 0.849, 0.373, 0.416, 0.57, 0.886, -0.284],
           [0.595, 0.97, -0.191, -0.395, -0.728, -0.976, -0.895, -0.95, -0.706, -0.688],
           [0.957, 0.751, 0.849, -0.728, 0.537, 0.456, -0.183, 0.705, -0.189, 0.985],
           [0.381, 0.7, 0.373, -0.976, 0.456, 0.808, -0.165, -0.553, -0.842, -0.448],
           [0.756, -0.179, 0.416, -0.895, -0.183, -0.165, 0.417, 0.392, -0.564, 0.527],
           [0.823, -0.161, 0.57, -0.95, 0.705, -0.553, 0.392, 0.428, -0.125, 0.958],
           [0.498, 0.438, 0.886, -0.706, -0.189, -0.842, -0.564, -0.125, -0.702, 0.105],
           [-0.164, -0.886, -0.284, -0.688, 0.985, -0.448, 0.527, 0.958, 0.105, 0.599]]

init_solution = greedy_naive_solution(test_w, test_n)

