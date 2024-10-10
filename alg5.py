import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
import os
from sklearn.preprocessing import StandardScaler

os.environ['LOKY_MAX_CPU_COUNT'] = '4'


def simulate_graph(G, n_dimensions, attraction_forces, num_steps=100, dt=0.01, mass=1.0, damping=0.9):
    positions = {}
    velocities = {}
    for node in G.nodes():
        positions[node] = np.random.uniform(0, 1, size=n_dimensions)
        velocities[node] = np.zeros(n_dimensions)

    for step in range(num_steps):
        forces = {node: np.zeros(n_dimensions) for node in G.nodes()}

        for (u, v), force_value in attraction_forces.items():
            pos_u = positions[u]
            pos_v = positions[v]
            delta_pos = pos_v - pos_u
            distance = np.linalg.norm(delta_pos) + 1e-6
            direction = delta_pos / distance
            force_vector = direction * force_value
            forces[u] += force_vector
            forces[v] -= force_vector

        for node in G.nodes():
            velocities[node] = damping * velocities[node] + (forces[node] / mass) * dt
            positions[node] += velocities[node] * dt

    return positions

def simulation(n, weights):
    G = nx.Graph()
    G.add_nodes_from(list(range(n)))
    edges = []
    for i in range(len(weights)):
        for j in range(i + 1, len(weights)):
            edges.append((i, j))
    attraction_forces = {
        k: weights[k[0]][k[1]] for k in edges
    }
    k = 2

    positions = simulate_graph(G, n_dimensions=k, attraction_forces=attraction_forces)
    print(positions)
    X = np.array(list(positions.values()))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    dbscan = DBSCAN(eps=0.4, min_samples=1)
    labels = dbscan.fit_predict(X_scaled)
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.title('Кластеризация методом k-средних')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    cluster_count = max(*labels) + 1
    cluster_labels = {key: int(label) for key, label in zip(positions.keys(), labels)}

    cliques = {i: list() for i in range(cluster_count)}

    for k, i in cluster_labels.items():
        if i != -1:
            cliques[i] += [k]
        else:
            cliques[len(cliques.keys())] = [k]

    balanced_cliques = []
    for vs in cliques.values():
        if len(vs) > 4:
            for v in vs:
                balanced_cliques.append([v])
        else:
            balanced_cliques.append(vs)
    return dict(zip(range(len(balanced_cliques)), balanced_cliques))

