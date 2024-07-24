import numpy as np
import scipy.sparse as spla
import scipy.sparse.linalg as splinalg
import networkx as ntx


def edges(layers):
    m1 = []
    for j in np.arange(0, 8 * layers, 8):
        m1.append([(i + j, i + 1 + j) for i in range(7)])
    return np.array(m1).reshape(-1, 2)

def edges2(layers):
    m2 = []
    for j in np.arange(0, 8 * layers, 8):
        m2.append(np.array([
            ((j + i) % (8 * layers), (j + 7 + i) % (8 * layers))  # Use modulo to wrap around
            for i in np.arange(1, 7, 2)
        ]))
    m2 = np.array(m2).reshape(-1, 2)
    return m2

def edges3(layers):
    m3 = []
    m3.append([(j + 7, j + 14) for j in np.arange(0, 8 * layers, 8)])
    m3 = np.array(m3).reshape(-1, 2)
    return m3

def create_graph(layers, seed):
    np.random.seed(seed)
    G = ntx.Graph()
    G.add_edges_from(edges(layers))
    G.add_edges_from(edges2(layers))
    G.add_edges_from(edges3(layers))
    return G

def generate_random_values(mean, std, seed, num_edges):
    np.random.seed(seed)
    return np.abs(np.random.normal(mean, std, num_edges))

# Cache for matrix_interactions results
MI_cache = {}

def matrix_interactions(layers):
    if layers in MI_cache:
        return MI_cache[layers]

    grid = create_graph(layers, 1)
    num_edges = grid.number_of_edges()
    nodes = grid.number_of_nodes()
    cap_values = generate_random_values(1, 0.01, 1, num_edges)

    adjM = spla.lil_matrix((nodes, nodes)) 
    xxx = list(grid.edges())
    interactions = 1 / cap_values

    for edge_idx, (i, j) in enumerate(xxx):
        if i < nodes and j < nodes:  # Check if indices are valid
            interaction_value = interactions[edge_idx]
            adjM[i, j] = interaction_value
            adjM[j, i] = interaction_value
        else:
            print(f"Warning: Skipping invalid edge ({i}, {j})")  # Optional warning

    adjM.setdiag(adjM.sum(axis=1).A1 - 0.0001)
    adjM = adjM.tocsc()  

    MI = splinalg.inv(adjM)
    MI_cache[layers] = MI
    return MI

def R(layers):
    MI = matrix_interactions(layers)
    m11 = MI[0, 0]
    m77 = MI[6, 6]
    return m11 + m77 - MI[1, 6] - MI[6, 1]

# Preallocate Rin array
layers_range = np.arange(3, 100, 1)
Rin = np.zeros(len(layers_range))

for idx, layers in enumerate(layers_range):
    print(layers)
    Rin[idx] = np.abs(R(layers))

plt.plot()