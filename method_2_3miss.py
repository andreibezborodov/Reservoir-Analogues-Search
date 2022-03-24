import numpy as np
import random
import time
import pandas as pd
from sklearn.manifold import TSNE
from utils import preprocess_data
from clustering import leiden_alg
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import warnings
from tqdm import tqdm

'''
    Метод 2: 30 запусков, 10 тестовых месторождения, в каждом по 3 параметра с пропусками - исследуем точности по одному
    Для каждого таргет параметра рандомно убираем два других
    Исключаем параметры с пропусками, находим кластера на всех параметрах, тренируем регрессию, чтобы предсказать
    кластера для таргетных месторождений в таргет параметре 
'''

warnings.filterwarnings('ignore')

parameters = ['TectonicRegime', 'HydrocarbonType', 'Period', 'DepositionalSystem', 'Lithology', 'TrappingMechanism',
            'StructuralSetting', 'Gross', 'Netpay', 'Porosity', 'Permeability', 'Depth']
cont_params = ['Gross', 'Netpay', 'Porosity', 'Permeability', 'Depth']
cat_params = ['TectonicRegime', 'HydrocarbonType', 'Period', 'DepositionalSystem', 'Lithology', 'TrappingMechanism',
            'StructuralSetting']

data = pd.read_csv("daks_new.csv", encoding='cp1251', delimiter=',')
x = data.loc[:, parameters].values
data = pd.DataFrame(data=x, columns=parameters)
data.dropna(inplace=True)
data.reset_index(inplace=True, drop=True)

dict_res = {}
times = []
all_results = []

for i in tqdm(range(1, 31)):
    target_indexes = [random.randint(0, 467) for x in range(10)]
    print('Target reservoir indexes:', target_indexes)
    print('iteration: {}'.format(i))

    for ind, target_parameter in enumerate(parameters):
        start = time.time()

        # Фрейм для получения кластеров с исключением тестовых месторождений
        data_copy = data.copy()
        # Фрейм для лог регрессии после получения кластеров
        data_copy2 = data.copy()

        # Убираем параметры, в которых пропуск + еще 2 в data_copy2 для обучения логистической регрессии (не можем их использовать в трейне)
        data_copy2.drop(target_parameter, axis=1, inplace=True)

        params_to_drop = random.choices(population=data_copy2.columns, k=2)
        data_copy2.drop(params_to_drop, axis=1, inplace=True)
        params = [i for i in cat_params if i != target_parameter and i != params_to_drop[0] and i != params_to_drop[1]]

        _, onehot_target, _ = preprocess_data(data_copy2, params, method='onehot', perform_kbins=False)

        original_table, preprocessed, code_dict = preprocess_data(data_copy, cat_params, method='label', perform_kbins=False)
        target_param_values = preprocessed[target_parameter].loc[target_indexes].values

        target_df_data_2 = preprocessed.loc[target_indexes]

        for index in target_indexes:
            preprocessed.drop(index, axis=0, inplace=True)
            preprocessed.reset_index(inplace=True, drop=True)


        target_df_data = onehot_target.loc[target_indexes]

        preprocessed_tsne = TSNE(n_components=3).fit_transform(preprocessed)

        graph, clustered_data = leiden_alg(preprocessed_tsne, metric='cosine', threshold=0.8)
        original_table['cluster'] = clustered_data
        preprocessed['cluster'] = clustered_data

        num_clusters = max(clustered_data)

        for index in target_indexes:
            onehot_target.drop(index, axis=0, inplace=True)
            onehot_target.reset_index(inplace=True, drop=True)

        y = preprocessed['cluster']
        X = onehot_target

        logmodel = LogisticRegression()
        logmodel.fit(X, y)

        predictions = logmodel.predict(target_df_data)

        target_df_data_2['cluster'] = predictions

        final_table = pd.concat([preprocessed, target_df_data_2], ignore_index=True)
        clustered_data = pd.concat([clustered_data, pd.Series(predictions)], ignore_index=True)

        Preds = []
        Tests = []
        results = {}

        for target_reservoir in range(468, 478):
            target_cluster = final_table.at[target_reservoir, 'cluster']
            cluster_indexes = np.where(clustered_data == target_cluster)

            for ind in cluster_indexes:
                table_with_target_cluster = final_table.loc[ind]

            X = pd.DataFrame(table_with_target_cluster)

            if target_parameter == 'Porosity' or target_parameter == 'Gross' or target_parameter == 'Netpay' \
                    or target_parameter == 'Depth' or target_parameter == 'Permeability':
                X = pd.DataFrame(table_with_target_cluster.drop(['cluster'], axis=1))

                Tests.append(X.loc[target_reservoir, target_parameter])

                X = X.drop(index=target_reservoir)
                X.reset_index(drop=True, inplace=True)

                Y_train = np.array(X.loc[:, target_parameter].values)

                Preds.append(np.mean(Y_train))
            else:
                X = pd.DataFrame(table_with_target_cluster.drop(['cluster'], axis=1))

                Tests.append(X.loc[target_reservoir, target_parameter])

                X = X.drop(index=target_reservoir)
                X.reset_index(drop=True, inplace=True)

                Y_train = X[target_parameter].values
                Preds.append(np.argmax(np.bincount(Y_train)))

        if target_parameter == 'Porosity' or target_parameter == 'Gross' or target_parameter == 'Netpay' \
                or target_parameter == 'Depth' or target_parameter == 'Permeability':
            ResultAcc = round(mean_squared_error(Tests, Preds, squared=False), 2)
            median_value = table_with_target_cluster[target_parameter].max() - table_with_target_cluster[target_parameter].min()
            FinalRes = ResultAcc / median_value

            print("{}: {}".format(target_parameter, FinalRes))

            results["{}".format(target_parameter)] = FinalRes
            dict_res["{}".format(target_parameter)] = FinalRes

            Tests = []
            Preds = []
            Y_train = []
        else:
            ResultAcc = 1 - round(accuracy_score(Tests, Preds), 2)
            print("{}: {}".format(target_parameter, ResultAcc))

            results["{}".format(target_parameter)] = ResultAcc
            dict_res["{}".format(target_parameter)] = ResultAcc

            Tests = []
            Preds = []
            Y_train = []

        original_table.drop('cluster', axis=1, inplace=True)
        preprocessed.drop('cluster', axis=1, inplace=True)

        end = time.time()
        times.append(end-start)
        # print(end - start)

    print(dict_res)
    all_results.append(dict_res)
    dict_res = {}

# print(times)
# print(np.mean(np.array(times)))

print('Printing Average Result Values:')
def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict
print(dict_mean(all_results))




