def remove_negative_vertices(weights):
    n = len(weights)
    vertex_sums = []
    for i in range(n):
        total = sum(weights[i][j] for j in range(n) if i != j)
        vertex_sums.append(total)

    if vertex_sums and max(vertex_sums) <= 0:
        return [min(*[enumerate(vertex_sums)], key=lambda x: x[1])[0]]
    else:
        return [i for i, s in enumerate(vertex_sums) if s > 0]


def vertex_deleter_solution(n, weights):
    vertexes = set(i for i in range(n))
    cl = []
    remaining_vertices = [1]
    vert_name_map = dict(zip(range(len(vertexes)), list(vertexes)))
    while len(remaining_vertices) != 0:
        remaining_vertices = remove_negative_vertices(weights)
        if remaining_vertices:
            cl.append(list(map(lambda x: vert_name_map[x], remaining_vertices)))
        weights = [[weights[i][j] for j in range(len(weights)) if j not in remaining_vertices] for i in
                   range(len(weights))
                   if i not in remaining_vertices]
        vertexes -= set(list(map(lambda x: vert_name_map[x], remaining_vertices)))
        vert_name_map = dict(zip(range(len(weights)), vertexes))
    return dict(zip(range(len(cl)), cl))



