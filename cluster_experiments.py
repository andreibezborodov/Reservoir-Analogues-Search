import numpy as np
import matplotlib.pyplot as plt
from regression import MultivariateRegression
from clustering import leiden_alg
from collections import Counter
from itertools import chain

def accuracy_with_diff_thresholds(dim_reduced_data, original_table):
    parameters = ['TectonicRegime', 'HydrocarbonType', 'Period', 'DepositionalSystem', 'Lithology', 'TrappingMechanism',
            'StructuralSetting', 'Gross', 'Netpay', 'Porosity', 'Permeability', 'Depth']
    overall_accuracy = []
    for trsh in np.arange(0.5, 0.9, 0.05):
        graph, clustered_data = leiden_alg(dim_reduced_data, threshold=trsh)
        original_table['cluster'] = clustered_data
        accuracy = MultivariateRegression(original_table, clustered_data, max(clustered_data))
        overall_accuracy.append(accuracy)

    print(overall_accuracy)

    for param in parameters:
        values = []
        for i in range(len(overall_accuracy)):
            current_dict = overall_accuracy[i]
            val = current_dict[param]
            values.append(val)

        plt.plot(np.arange(0.5, 0.9, 0.05), values)
        plt.xlabel('Cosine dist threshold')
        plt.ylabel('Parameter Accuracy')
        plt.title('{} Accuracy'.format(param))
        plt.show()


def find_same_reservoirs_in_diff_clusters(num_of_networks, dim_reduced_data, original_table):
    dict1 = {}
    dict2 = {}
    dict3 = {}
    # dict4 = {}
    # dict5 = {}

    graph, clustered_data = leiden_alg(dim_reduced_data, threshold=0.9)
    reservoirs1 = list(clustered_data)
    cluster_indexes1 = list(set(reservoirs1))

    graph, clustered_data2 = leiden_alg(dim_reduced_data, threshold=0.8)
    reservoirs2 = list(clustered_data2)
    cluster_indexes2 = list(set(reservoirs2))

    graph, clustered_data3 = leiden_alg(dim_reduced_data, threshold=0.7)
    reservoirs3 = list(clustered_data3)
    cluster_indexes3 = list(set(reservoirs3))

    # graph, clustered_data4 = leiden_alg(data_transfoemed, threshold=0.6)
    # reservoirs4 = list(clustered_data4)
    # cluster_indexes4 = list(set(reservoirs4))
    #
    # graph, clustered_data5 = leiden_alg(data_transfoemed, threshold=0.5)
    # reservoirs5 = list(clustered_data5)
    # cluster_indexes5 = list(set(reservoirs5))

    for cluster in cluster_indexes1:
        indices1 = [i for i, x in enumerate(reservoirs1) if x == cluster]
        dict1[cluster] = indices1

    for cluster in cluster_indexes2:
        indices2 = [i for i, x in enumerate(reservoirs2) if x == cluster]
        dict2[cluster] = indices2

    for cluster in cluster_indexes3:
        indices3 = [i for i, x in enumerate(reservoirs3) if x == cluster]
        dict3[cluster] = indices3

    # for cluster in cluster_indexes4:
    #     indices4 = [i for i, x in enumerate(reservoirs4) if x == cluster]
    #     dict4[cluster] = indices4
    #
    # for cluster in cluster_indexes5:
    #     indices5 = [i for i, x in enumerate(reservoirs5) if x == cluster]
    #     dict5[cluster] = indices5

    occurencies = {}
    i = 1

    for cluster1 in dict1.keys():
        for reservoir in dict1[cluster1]:
            for cl2, res2 in dict2.items():
                if reservoir in res2:
                    target_cl2 = cl2
                    break
            for cl3, res3 in dict3.items():
                if reservoir in res3:
                    target_cl3 = cl3
                    break
            # for cl4, res4 in dict4.items():
            #     if reservoir in res4:
            #         target_cl4 = cl4
            #         break
            # for cl5, res5 in dict5.items():
            #     if reservoir in res5:
            #         target_cl5 = cl5
            #         break
            lists = [dict1[cluster1], dict2[target_cl2], dict3[target_cl3]]

            no_of_lists_per_name = Counter(chain.from_iterable(map(set, lists)))

            for name, no_of_lists in no_of_lists_per_name.most_common():
                if no_of_lists == 2:
                    occurencies[reservoir] = i
                    i += 1

            i = 1
            # counter = collections.Counter(sum(lists))
            # occurencies[reservoir] = counter.most_common()

            # for oc_res in dict1[cluster1]:
            #     if oc_res in dict2[target_cl2]:
            #         if oc_res in dict3[target_cl3]:
            #             if oc_res in dict4[target_cl4]:
            #                 if oc_res in dict5[target_cl5]:
            #                     occurencies[oc_res] = i
            #                     i += 1

    print(occurencies)

    # sns.displot(unique_values, bins=50, kde=True)
    # plt.title('Distribution with 0.5-0.9 threshold', fontsize=18)
    # plt.xlabel('Number of close analogues', fontsize=16)
    # plt.ylabel('Frequency', fontsize=16)
    # plt.show()

    print(dict1)
    print(dict2)
    print(dict3)

