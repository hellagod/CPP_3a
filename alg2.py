from setuptools.config.pyprojecttoml import load_file


def remove_negative_vertices(weights):
    n = len(weights)
    vertex_sums = []
    for i in range(n):
        total = sum(weights[i][j] for j in range(n) if i != j)
        vertex_sums.append(total)

    if vertex_sums and max(vertex_sums) <=0:
        return [min(*[enumerate(vertex_sums)],key=lambda x: x[1])[0]]
    else:
        return [i for i, s in  enumerate(vertex_sums) if s > 0]

weights = [[3.0, 7.0, 8.0, 2.0, 1.0, 8.0, 10.0, -1e+20, 4.0, 5.0], [7.0, 7.0, 10.0, 5.0, 7.0, 7.0, 3.0, 10.0, 6.0, 2.0], [8.0, 10.0, -1e+20, -1e+20, -1e+20, 6.0, 3.0, 6.0, 1.0, 6.0], [2.0, 5.0, -1e+20, 3.0, 3.0, -1e+20, 9.0, 5.0, -1e+20, 8.0], [1.0, 7.0, -1e+20, 3.0, 8.0, -1e+20, -1e+20, 4.0, 5.0, 4.0], [8.0, 7.0, 6.0, -1e+20, -1e+20, 7.0, 3.0, 2.0, 4.0, 9.0], [10.0, 3.0, 3.0, 9.0, -1e+20, 3.0, -1e+20, 4.0, 8.0, 5.0], [-1e+20, 10.0, 6.0, 5.0, 4.0, 2.0, 4.0, 10.0, -1e+20, -1e+20], [4.0, 6.0, 1.0, -1e+20, 5.0, 4.0, 8.0, -1e+20, -1e+20, 8.0], [5.0, 2.0, 6.0, 8.0, 4.0, 9.0, 5.0, -1e+20, 8.0, 8.0]]

def vertex_deleter(n, weights):
    vertexes = set(i for i in range(n))
    cl = []
    remaining_vertices = [1]
    vert_name_map =  dict(zip(range(len(vertexes)), list(vertexes)))
    while len(remaining_vertices) != 0:
        remaining_vertices = remove_negative_vertices(weights)
        if remaining_vertices:
            cl.append(list(map(lambda x: vert_name_map[x], remaining_vertices)))
        weights = [[weights[i][j] for j in range(len(weights)) if j not in remaining_vertices] for i in range(len(weights))
                   if i not in remaining_vertices]
        vertexes -= set(list(map(lambda x: vert_name_map[x], remaining_vertices)))
        vert_name_map = dict(zip(range(len(weights)), vertexes))
    return dict(zip(range(len(cl)), cl))

print(vertex_deleter(10, weights))