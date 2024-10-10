def clique_weight(vertices_list, weights):
    total_weight = 0
    for vertices in vertices_list:
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                total_weight += weights[vertices[i]][vertices[j]]
    return total_weight