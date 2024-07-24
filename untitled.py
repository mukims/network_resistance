import numpy as np
import scipy.sparse as spla
import scipy.sparse.linalg as splinalg
import networkx as ntx

def E0(layers):
    m1=[]
    for j in np.arange(0,8*layers,8):
        m1.append([(i+j,i+1+j) for i in range(7)])
    return np.array(m1).reshape(-1,2)
def E1(layers):
    m2=[]
    for j in np.arange(0,8*(layers-1),8):
        m2.append(np.array([((j+i,j+7+i),(j+i,j+7+2+i)) for i in np.arange(1,7,2)]))
    m2=np.array(m2).reshape(-1,2)
    return m2
def E2(layers):
    m3=[]
    m3.append([(j+7,j+14) for j in np.arange(0,8*(layers-1),8)])
    m3=np.array(m3).reshape(-1,2)
    return m3
def E3(layers):
    m4=[]
    m4.append([(8*layers+1,i) for i in np.arange(0,7,2)])
    return np.array(m4).reshape(-1,2)


def E4(layers):
    m5=[]
    m5.append([(layers*8,i) for i in np.arange((layers-1)*8+1,layers*8,2)])
    return np.array(m5).reshape(-1,2)
def create_graph(layers,seed):
    np.random.seed(seed)
    number_of_indices = 8*layers
    G = ntx.Graph()
    #G.add_nodes_from(range(8 * layers))
    e0 = E0(layers)
    e1 = E1(layers)
    e2 = E2(layers)
    e3 = E3(layers)
    e4 = E4(layers)
    G.add_edges_from(e0)
    G.add_edges_from(e1)
    G.add_edges_from(e2)
  #  G.add_edges_from(e3)
  #  G.add_edges_from(e4)
#    G.add_edges_from(((0,11),(6,20)))
    return G

G = create_graph(4,1)
pos = ntx.spring_layout(G)
# Draw graph with node labels
ntx.draw(G, pos, with_labels=True, node_size=100, font_size=5)

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
    edges2 = list(grid.edges())
    interactions = 1 / cap_values

    for edge_idx, (i, j) in enumerate(edges2):
        interaction_value = interactions[edge_idx]
        adjM[i, j] = interaction_value
        adjM[j, i] = interaction_value

    adjM.setdiag(adjM.sum(axis=1).A1 - 0.0001)
    adjM = adjM.tocsc()  

    MI = splinalg.inv(adjM)
    MI_cache[layers] = MI
    return MI

def R(layers):
    MI = matrix_interactions(layers)
    m11 = MI[1, 1]
    m77 = MI[7, 7]
    return m11 + m77 - MI[1, 7] - MI[7, 1]

# Preallocate Rin array
layers_range = np.arange(3, 100, 1)
Rin = np.zeros(len(layers_range))

for idx, layers in enumerate(layers_range):
    print(layers)
    Rin[idx] = np.abs(R(layers))
