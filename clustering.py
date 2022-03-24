from utils import create_cosine_graph, create_hamming_distance_graph, create_heom_distance_graph, \
    create_hvdm_distance_graph, create_gower_graph
import numpy as np
import pandas as pd
from cdlib import algorithms
from sklearn.cluster import DBSCAN
import community as community_louvain
import igraph
import leidenalg
import model

def eigenvector_clusteriser(vectors: pd.Series, threshold=None):
    g, adj_matrix = create_cosine_graph(vectors, threshold)
    comms = algorithms.eigenvector(g)
    comms = comms.communities
    comms = dict(sorted({item: list_id for list_id, l in enumerate(comms) for item in l}.items()))
    return g, pd.Series(comms)


def dbscan_clusteriser(vectors: pd.Series, eps=0.25, min_samples=8):
    vectors = np.stack(np.array(vectors))
    model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(vectors)
    return pd.Series(model.labels_)


def louvain_clusteriser(vectors: pd.Series, threshold=None):
    g, adj_matrix = create_cosine_graph(vectors, threshold)
    partition = community_louvain.best_partition(g)
    return g, pd.Series(list(partition.values()))

def leiden_alg(vectors: pd.Series, metric:str, threshold):
    if metric == 'cosine':
        gr, adj_matrix = create_cosine_graph(vectors, threshold)
    elif metric == 'gower':
        gr, adj_matrix = create_gower_graph(vectors, threshold)
    elif metric == 'hamming':
        gr, adj_matrix = create_hamming_distance_graph(vectors, threshold)
    elif metric == 'heom':
        gr, adj_matrix = create_heom_distance_graph(vectors, threshold)
    elif metric == 'hvdm':
        gr, adj_matrix = create_hvdm_distance_graph(vectors, threshold)
    else:
        raise Exception('Incorrect metric (Possible choices: cosine, gower, hamming, heom, hvdm')
    # g2 = igraph.Graph(len(g), list(zip(*list(zip(*nx.to_edgelist(g)))[:2])))

    weights = []

    g = igraph.Graph(directed=False)
    g.add_vertices(vectors.shape[0])
    for i in range(vectors.shape[0]):
        for j in range(i):
            if (adj_matrix[i, j] != 0):
                g.add_edges([(i, j)])
                weights.append(adj_matrix[i, j])

    partition = leidenalg.find_partition(g, partition_type=leidenalg.ModularityVertexPartition, n_iterations=50,
                                         weights=weights)

    # partition = leidenalg.CPMVertexPartition(g, resolution_parameter=0.1)
    # optimiser = leidenalg.Optimiser()
    # diff = optimiser.optimise_partition(partition)
    return gr, pd.Series(partition.membership)


def leiden_wbfm(vectors: pd.Series, attributes: pd.Series, metric:str, threshold):
    if metric == 'cosine':
        gr, adj_matrix = create_cosine_graph(vectors, threshold)
        gr_attr, adj_matrix_attr = create_cosine_graph(attributes, threshold)
    elif metric == 'gower':
        gr, adj_matrix = create_gower_graph(vectors, threshold)
    elif metric == 'hamming':
        gr, adj_matrix = create_hamming_distance_graph(vectors, threshold)
    elif metric == 'heom':
        gr, adj_matrix = create_heom_distance_graph(vectors, threshold)
    elif metric == 'hvdm':
        gr, adj_matrix = create_hvdm_distance_graph(vectors, threshold)
    else:
        raise Exception('Incorrect metric (Possible choices: cosine, gower, hamming, heom, hvdm')
    # g2 = igraph.Graph(len(g), list(zip(*list(zip(*nx.to_edgelist(g)))[:2])))

    weights = []

    for e in gr.edges():
        gr[e[0]][e[1]]['weight'] = adj_matrix[e[0], e[1]]

    for e in gr_attr.edges():
        gr_attr[e[0]][e[1]]['weight'] = adj_matrix_attr[e[0], e[1]]

    rho = 0.25

    partition, metrics_report = model.leiden_wbfm(gr, gr_attr, rho=rho)
    return gr, gr_attr, partition