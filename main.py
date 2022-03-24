import argparse
import pandas as pd
from sklearn.manifold import TSNE
from utils import preprocess_data
from clustering import leiden_alg, leiden_wbfm, louvain_clusteriser, eigenvector_clusteriser
from visualization import visualize_network
from regression import MultivariateRegression

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

parameters = ['TectonicRegime', 'HydrocarbonType', 'Period', 'DepositionalSystem', 'Lithology', 'TrappingMechanism',
            'StructuralSetting', 'Gross', 'Netpay', 'Porosity', 'Permeability', 'Depth']

cont_params = ['Gross', 'Netpay', 'Porosity', 'Permeability', 'Depth']
disc_params = ['TectonicRegime', 'HydrocarbonType', 'Period', 'DepositionalSystem', 'Lithology', 'TrappingMechanism',
            'StructuralSetting']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--approach",
        help="choose baseline, Louvain, Newman, Leiden or Leiden WBFM Algorithm for clustering",
        default='louvain',
        choices=['leiden', 'leiden_attr', 'baseline', 'newman', 'louvain'],
        type=str
    )
    parser.add_argument(
        "--metric",
        help="choose the metric for creating a graph for analysis",
        default='cosine',
        choices=['cosine', 'gower', 'hamming', 'heom', 'hvdm'],
        type=str
    )
    parser.add_argument(
        "--threshold",
        help="choose the threshold to perform Leiden Alg reservoirs clustering",
        default=0.9
    )
    parser.add_argument(
        "--visualize",
        help="plot the graph with reservoirs of different colors",
        default=False
        )
    args = parser.parse_args()

    approach = str(args.approach)
    metric = str(args.metric)
    threshold = float(args.threshold)
    visualize = args.visualize

    data = pd.read_csv("daks_new.csv", encoding='cp1251', delimiter=',')
    x = data.loc[:, parameters].values
    data = pd.DataFrame(data=x, columns=parameters)
    data.dropna(inplace=True)
    data.reset_index(inplace=True, drop=True)

    # choose the method for categorical data encoding: 'onehot' or 'label'
    # choose whether to perform the discretization of continuous data or not
    original_table, preprocessed, code_dict = preprocess_data(data, disc_params, method='label', perform_kbins=False)

    print(preprocessed.to_string())

    if approach == 'leiden':
        preprocessed = TSNE(n_components=3).fit_transform(preprocessed)
        graph, clustered_data = leiden_alg(preprocessed, metric=metric, threshold=threshold)
        original_table['cluster'] = pd.Series(clustered_data)
        print(original_table['cluster'].max())
        clusters = original_table['cluster'].tolist()
    elif approach == 'baseline':
        x = StandardScaler().fit_transform(preprocessed)
        pca = PCA(n_components=5)
        principalComponents = pca.fit_transform(x)
        clustering = KMeans(n_clusters=7, random_state=0).fit(principalComponents)
        original_table['cluster'] = pd.Series(clustering.labels_)
        clusters = original_table['cluster'].tolist()
    elif approach == 'leiden_attr':
        preprocessed_cat = preprocessed[disc_params]
        preprocessed_cont = preprocessed[cont_params]

        preprocessed_attr = TSNE(n_components=3).fit_transform(preprocessed_cat)
        preprocessed = TSNE(n_components=3).fit_transform(preprocessed)

        graph, gr_attr, clustered_data = leiden_wbfm(preprocessed, preprocessed_attr, metric=metric,
                                                     threshold=threshold)
        original_table['cluster'] = pd.Series(clustered_data)
        clusters = original_table['cluster'].tolist()
    elif approach == 'newman':
        preprocessed = TSNE(n_components=3).fit_transform(preprocessed)
        graph, clustered_data = eigenvector_clusteriser(preprocessed, threshold=0.9)
        original_table['cluster'] = pd.Series(clustered_data)
        print(original_table['cluster'].max())
        clusters = original_table['cluster'].tolist()
    elif approach == 'louvain':
        preprocessed = TSNE(n_components=3).fit_transform(preprocessed)
        graph, clustered_data = louvain_clusteriser(preprocessed, threshold=0.9)
        original_table['cluster'] = pd.Series(clustered_data)
        print(original_table['cluster'].max())
        clusters = original_table['cluster'].tolist()
    else:
        raise NameError("Not supported algorithm. Possible choices = leiden, leiden_attr, newman, louvain, baseline")

    if visualize:
        # TODO generalize visualization
        visualize_network(clustered_data, graph, 'graph_newman_2.pdf')
        # visualize_network(clustered_data, gr_attr, 'graph_attr.png')

    MultivariateRegression(original_table, clusters, max(clusters))