import pulp
import random
import networkx as nx
from matplotlib import pyplot as plt


def naive_solution(n):
    return {(i, j): 1 for i in range(n) for j in range(i + 1, n)}


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
        for (i, j, w) in clique:
            naive_sol[(i, j)] = 1
    return naive_sol


class CPPSoftModel:
    def __init__(self, n, weights, soft=True):
        self.weights = weights
        self.n = n
        self.soft = soft

        self.graph = self._create_graph()
        self.model = pulp.LpProblem("cpp", pulp.LpMaximize)

        self._init_vars()
        self._init_objectives()
        self._init_constraints()

        if self.soft:
            naive_sol = greedy_naive_solution(n, weights)
            for (i, j) in naive_sol:
                if (i, j) in self.x:
                    self.x[(i, j)].setInitialValue(naive_sol[(i, j)])

    def _create_graph(self):
        graph = nx.complete_graph(self.n)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                graph[i][j]['weight'] = self.weights[i][j]
        return graph

    def _init_vars(self):
        n = self.n
        self.x = pulp.LpVariable.dicts("x", ((i, j) for i in range(n) for j in range(i + 1, n)), lowBound=0, upBound=1,
                                       cat=pulp.LpInteger)

    def _init_objectives(self):
        x, n, graph = self.x, self.n, self.graph
        self.model += pulp.lpSum(
            graph[i][j]['weight'] * x[(i, j)] for i in range(n) for j in range(i + 1, n)), "total_weight"

    def _init_constraints(self):
        model, x, n = self.model, self.x, self.n

        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    model += (x[(i, j)] + x[(j, k)] - x[(i, k)] <= 1, f"constraint_1_{i}_{j}_{k}")
                    model += (x[(i, j)] - x[(j, k)] + x[(i, k)] <= 1, f"constraint_2_{i}_{j}_{k}")
                    model += (-x[(i, j)] + x[(j, k)] + x[(i, k)] <= 1, f"constraint_3_{i}_{j}_{k}")

    def just_solve(self):
        self.model.solve(pulp.PULP_CBC_CMD(msg=False, warmStart=self.soft))

    def solve(self):
        model = self.model
        model.solve(pulp.PULP_CBC_CMD(msg=False, warmStart=self.soft))
        self.clique_edges = [tuple(map(int, x.name.replace('x_(', '').replace(')', '').split(',_'))) for x in
                             model.variables() if x.varValue]
        graph = nx.Graph()
        graph.add_edges_from(self.clique_edges)
        self.connected_components = list(nx.connected_components(graph))
        self.separate_graphs = [list(c) for c in self.connected_components]

    def output(self):
        model = self.model
        f = model.objective.value()

        return {
            'f': f,
            'cliques': self.separate_graphs,
            'clique_edges': self.clique_edges
        }

    def plot(self):
        pos = nx.circular_layout(self.graph)

        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        colors = ["#" + ''.join([random.choice('789ABCDEF') for j in range(6)])
                  for i in range(len(self.separate_graphs))]
        edge_colors = []

        for i, j in self.graph.edges():
            if (i, j) in self.clique_edges or (j, i) in self.clique_edges:
                for k, g in enumerate(self.separate_graphs):
                    if i in g:
                        edge_colors.append(colors[k])
            else:
                edge_colors.append('lightgray')

        nx.draw(self.graph, pos, with_labels=True, node_color='lightblue', node_size=500, edge_color=edge_colors,
                width=2)
        nx.draw_netwoarkx_edge_labels(self.graph, pos, edge_labels=edge_labels)

        plt.title("Оптимальное разбиение графа на клики")
        plt.savefig('./imgs/clique_graph_m_2.png')
        plt.show()





