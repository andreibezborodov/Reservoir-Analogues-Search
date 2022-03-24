from leidenutils import *
import community


def leiden_wbfm(G_structure, G_attributes, rho=0.5):

    structure_partition = community.best_partition(G_structure)
    attributes_partition = community.best_partition(G_attributes)
    structure_modularity = community.modularity(structure_partition, G_structure)
    attributes_modularity = community.modularity(attributes_partition, G_attributes)

    alpha = rho * attributes_modularity / (rho * attributes_modularity + (1 - rho) * structure_modularity)

    metrics_report, clusters, G_w, partition = calculate_metrics_graphs(G_structure, G_attributes,
                                                                     algo='leiden', alpha=alpha)

    return partition, metrics_report