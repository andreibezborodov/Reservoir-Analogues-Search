import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.preprocessing import KBinsDiscretizer
from scipy.spatial.distance import hamming
import gower
from distython import HEOM, HVDM

def code_categories(data: pd.DataFrame, method: str, columns: list) -> (pd.DataFrame, dict):
    data = data.dropna()
    data.reset_index(inplace=True, drop=True)
    d_data = data.copy()
    encoder_dict = dict()
    if method == 'label':
        for column in columns:
            le = LabelEncoder()
            d_data[column] = le.fit_transform(d_data[column].values)
            mapping = dict(zip(le.classes_, range(len(le.classes_))))
            encoder_dict[column] = mapping
    elif method == 'onehot':
        d_data = pd.get_dummies(d_data, columns=columns)
    return d_data, encoder_dict

def preprocess_data(data: pd.DataFrame, parameters:list, method:str, perform_kbins:bool):
    # x = data.loc[:, parameters].values
    # param_data = pd.DataFrame(data=x, columns=parameters)
    # param_data.dropna(inplace=True)
    # param_data.reset_index(inplace=True, drop=True)
    new_data, code_dict = code_categories(data, method, parameters)

    if perform_kbins:
        kbd = KBinsDiscretizer(encode='ordinal', strategy='quantile')
        parameters = ['Depth', 'Porosity', 'Gross', 'Netpay', 'Permeability']

        for param in parameters:
            new_data[param] = kbd.fit_transform(new_data[[param]])

    return data, new_data, code_dict

def create_cosine_graph(vectors: pd.Series, threshold=None):
    if isinstance(vectors, pd.DataFrame):
        vector_list = vectors.values.tolist()
    else:
        vector_list = vectors.tolist()
    W_sparse = csr_matrix(np.asarray(vector_list))
    cos_sim = cosine_similarity(W_sparse)
    if threshold == None:
        threshold = np.quantile(cos_sim.flatten(), [0.9])[0]
    # print(threshold)
    adj_matrix = (cos_sim > threshold).astype(int)
    adj_matrix = adj_matrix * cos_sim
    g = nx.convert_matrix.from_numpy_array(adj_matrix)
    # print(nx.info(g))
    return g, adj_matrix

def create_gower_graph(vectors: pd.Series, threshold=None):
    # vector_list = vectors.values.tolist()
    # W_sparse = csr_matrix(np.asarray(vector_list))
    weights = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    gow_sim = gower.gower_matrix(vectors, weight=weights)
    if threshold == None:
        threshold = np.quantile(gow_sim.flatten(), [0.9])[0]
    print(threshold)
    adj_matrix = (gow_sim > threshold).astype(int)
    adj_matrix = adj_matrix * gow_sim
    g = nx.convert_matrix.from_numpy_array(adj_matrix)
    print(nx.info(g))
    return g, adj_matrix

def create_hamming_distance_graph(vectors: pd.Series, threshold=None):
    # vector_list = vectors.tolist()
    vector_list = vectors.values.tolist()

    matrix = np.zeros((478, 478))
    distances = []
    k = 0

    for i in vector_list:
        for j in vector_list:
            dist = hamming(i, j)
            distances.append(dist)

        matrix[k, :] = distances
        k += 1
        distances = []

    print(threshold)
    adj_matrix = (matrix > threshold).astype(int)
    adj_matrix = adj_matrix * matrix
    g = nx.convert_matrix.from_numpy_array(adj_matrix)
    print(nx.info(g))
    return g, adj_matrix

def create_heom_distance_graph(vectors: pd.Series, threshold=None):
    # Label encoding
    # Need indexes of disc params
    vector_list = vectors.values
    cat_index = [0, 1, 2, 3, 4, 5, 6]
    heom_metric = HEOM(vector_list, cat_ix=cat_index)

    matrix = np.zeros((478, 478))
    distances = []
    k = 0

    for i in vector_list:
        for j in vector_list:
            i = np.asarray(i)
            i = i.astype(np.float64)
            j = np.asarray(j)
            j = j.astype(np.float64)
            dist = heom_metric.heom(i, j)
            distances.append(dist)

        matrix[k, :] = distances
        k += 1
        distances = []

    print(threshold)
    adj_matrix = (matrix > threshold).astype(int)
    adj_matrix = adj_matrix * matrix
    g = nx.convert_matrix.from_numpy_array(adj_matrix)
    print(nx.info(g))
    return g, adj_matrix

def create_hvdm_distance_graph(vectors: pd.Series, threshold=None):
    # Label encoding
    # Need indexes of disc params
    vector_list = vectors.values
    cat_index = [0, 1, 2, 3, 4, 5, 6]
    hvdm_metric = HVDM(vector_list, y_ix = [[cat_index]], cat_ix=cat_index)

    matrix = np.zeros((478, 478))
    distances = []
    k = 0

    for i in vector_list:
        for j in vector_list:
            i = np.asarray(i)
            i = i.astype(np.float64)
            j = np.asarray(j)
            j = j.astype(np.float64)
            dist = hvdm_metric.hvdm(i, j)
            distances.append(dist)

        matrix[k, :] = distances
        k += 1
        distances = []

    print(threshold)
    adj_matrix = (matrix > threshold).astype(int)
    adj_matrix = adj_matrix * matrix
    g = nx.convert_matrix.from_numpy_array(adj_matrix)
    print(nx.info(g))
    return g, adj_matrix