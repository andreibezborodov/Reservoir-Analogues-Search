import networkx as nx
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from pyvis.network import Network


def visualize_network(clustered_data, graph, figname):
    color_map = []
    colors = ['b', 'g', 'r', 'c', 'm', 'lime', 'y', 'orangered']
    i = 0
    #  'lime', 'y', 'orangered'
    for node in graph:
        color_index = clustered_data[i]
        color_map.append(colors[color_index])
        i += 1
    fig = plt.figure()

    ax = fig.subplots()
    nx.draw(graph, node_color=color_map, with_labels=False, node_size=50, width=0.2, ax=ax)
    # ax.set_title('Graph with clusters of different colors')
    # plt.show()
    fig.savefig(figname)

    # nt = Network('1000px', '1000px')
    # nt.from_nx(graph)
    # nt.show('nx.html')

