import numpy as np
import networkx as nx


def get_symptom_network(group_size=6, bridge_size=3):
    """
    Creates a network with two fully connected groups and a fully connected bridge between them.
    """
    n = group_size*2 + bridge_size
    adj_matrix = np.zeros((n, n))
    # group A
    for i in range(group_size):
        for j in range(group_size + bridge_size):
            adj_matrix[i, j] = 1
    # bridges
    for i in range(group_size, group_size + bridge_size):
        for j in range(n):
            adj_matrix[i, j] = 1
    # group B
    for i in range(group_size + bridge_size, n):
        for j in range(group_size, n):
            adj_matrix[i, j] = 1
    G = nx.from_numpy_array(adj_matrix, create_using=nx.Graph())
    G.remove_edges_from(nx.selfloop_edges(G))
    return G