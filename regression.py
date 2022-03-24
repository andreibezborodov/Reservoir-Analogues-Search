import numpy as np
import pandas as pd
from utils import code_categories
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score


def MultivariateRegression(table, clusters, numclusters):
    parameters = ['Depth', 'Gross', 'Netpay', 'Permeability', 'Porosity', 'TectonicRegime', 'HydrocarbonType',
                  'StructuralSetting', 'Period', 'Lithology', 'DepositionalSystem', 'TrappingMechanism']
    cat_params = ['TectonicRegime', 'HydrocarbonType', 'StructuralSetting', 'Period', 'Lithology', 'DepositionalSystem',
                  'TrappingMechanism']
    Preds = []
    Tests = []

    results = {}

    # for ind, el in enumerate(clusters):
    #     if el > 6:
    #         clusters[ind] = 1

    for param in parameters:
        for ClusterID in range(numclusters):
            TargetCluster = ClusterID
            clusters = np.array(clusters)
            Indexes = np.where(clusters == TargetCluster)

            for ind in Indexes:
                TableWithClusters = table.loc[ind]

            X = pd.DataFrame(TableWithClusters)
            for ind in range(X.shape[0]):
                if param == 'Porosity' or param == 'Gross' or param == 'Netpay' or param == 'Depth' or param == 'Permeability':
                    X = pd.DataFrame(TableWithClusters.drop(['cluster'], axis=1))
                    X.reset_index(drop=True, inplace=True)
                    Tests.append(X.loc[ind, param])
                    X = X.drop(index=ind)
                    X.reset_index(drop=True, inplace=True)
                    Y_train = np.array(X.loc[:, param].values)

                    Preds.append(np.mean(Y_train))
                else:
                    X = pd.DataFrame(TableWithClusters.drop(['cluster'], axis=1))
                    X, code_dict = code_categories(X, 'label', cat_params)
                    X.reset_index(drop=True, inplace=True)

                    Tests.append(X.loc[ind, param])

                    X = X.drop(index=ind)
                    X.reset_index(drop=True, inplace=True)

                    Y_train = X[param].values
                    Preds.append(np.argmax(np.bincount(Y_train)))

        if param == 'Porosity' or param == 'Gross' or param == 'Netpay' or param == 'Depth' or param == 'Permeability':
            ResultAcc = round(mean_squared_error(Tests, Preds, squared=False), 2)
            median_value = table[param].max() - table[param].min()
            FinalRes = ResultAcc / median_value
            print("{}: {}".format(param, FinalRes))

            results["{}".format(param)] = FinalRes

            Tests = []
            Preds = []
        else:
            ResultAcc = 1 - round(accuracy_score(Tests, Preds), 2)
            print("{}: {}".format(param, ResultAcc))

            results["{}".format(param)] = ResultAcc

            Tests = []
            Preds = []
    return results