import networkx as nx
from networkx.algorithms.shortest_paths import weighted
from networkx.classes.function import degree
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power

def combination(alist, N, repetition=True):
    if N==1:
        for i in alist:
            yield [i]
    else:
        for i in alist:
            if not repetition:
                templist = alist.copy()
                templist.remove(i)
            else:
                templist = alist
            for j in combination(templist, N-1):
                yield [i, *j]


def time_step_to_graph(seqformat=[1,1,0,0], tau=1, normalise=True):
    seqformat = np.array(seqformat)
    node_idx = list(np.nonzero(seqformat)[0])
    G = nx.Graph(name='G')
    for idx, i in enumerate(seqformat):
        G.add_node(i, name=f"time_{idx}")
    
    for edge in combination(node_idx, 2, repetition=True):
        if tau:
            G.add_edge(edge, weight= np.exp(np.abs(edge[0]-edge[1])))
        else:
            G.add_edge(edge)
    adj_norm=""
    return adj_norm

def seq_to_graph(seqformat=[1,1,0,0], tau_weighted=1, normalise=True):
    seq_len = len(seqformat)
    adj = np.zeros(shape=(seq_len,seq_len))
    degree = np.zeros(shape=(seq_len,seq_len))

    for i in range(seq_len):
        for j in range(seq_len):
            if seqformat[i]==1 and seqformat[j]==1:
                adj[i,j] = adj[j,i] = np.exp(-np.abs(i-j)/tau_weighted) if tau_weighted else 1
            if i==j:
                adj[i , j] = 1
    for i in range(seq_len):
        degree[i,i] = np.sum(adj[i])
    
    if normalise:
        d_half_norm= fractional_matrix_power(degree, -0.5)
        return d_half_norm.dot(adj).dot(d_half_norm)
    else:
        return adj, degree
                
