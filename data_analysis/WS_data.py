import datetime

import configparser
import os
import numpy as np
import pandas as pd

from data_analysis import ahp_util

config = configparser.ConfigParser()
parent = os.path.dirname
config.read(os.path.join(parent(parent(__file__)), 'config.ini'))
data_url = config['DEFAULT']['data_url']

numeric_data = 'Bodem en Watersturend WS v2_May 30, 2023_11.15'
cluster = 'WS'

data = ahp_util.read_survey_data(numeric_data, datetime.date(2023, 5, 23))

weights = pd.DataFrame(columns=list('ABCDEFGHIJKLMNOPQRSTUVWXYZab'))
for index, expert in data.iterrows():
    df_expert = data.iloc[[index-2]]  # -2 is hardcoded since index 2 is first row (rest is header
    df_expert = df_expert.dropna(axis=1, how='all')
    matrix = ahp_util.build_matrix(df_expert)
    points = list(df_expert.columns[-10].split('_')[1] + df_expert.columns[-9].split('_')[1][1] + df_expert.columns[-1].split('_')[1])
    weight_dict = dict.fromkeys(list('ABCDEFGHIJKLMNOPQRSTUVWXYZab'), 0)
    expertweights = ahp_util.compute_priority_weights(np.squeeze(matrix))
    for point, exp_weight in zip(points, expertweights):
        weight_dict[point] = exp_weight
    print(points, expertweights)
    new_df = pd.DataFrame(weight_dict, index=[0])
    weights = pd.concat([weights, new_df], ignore_index=True)
    print(ahp_util.compute_consistency_ratio(np.squeeze(matrix)))

# weights = weights.replace(0, np.NaN)
mean_weight = weights.replace(0, np.NaN).mean(axis=0).replace(np.NaN,0)
mean_std = weights.replace(0, np.NaN).std(axis=0).replace(np.NaN,0)
print(mean_weight)

dp, X, Y = ahp_util.read_dp_csv(cluster)

with open(data_url + "/expertscores_{}.csv".format(cluster), 'w') as f:
    f.write("Point,X,Y,Mean,Std\n")
    for p, x, y, mean, std in zip(dp, X, Y, mean_weight, mean_std):
        f.write("%s,%s,%s,%s,%s\n" % (p, x, y, mean, std))

# matrix = ahp_util.build_matrix(data)
# consistency_ratio = [ahp_util.compute_consistency_ratio(matrix[expert]) for expert in range(len(data))]
# priority_weights = [ahp_util.compute_priority_weights(matrix[expert]) for expert in range(len(data))]

# dp, X, Y = ahp_util.read_dp_csv(cluster)

# with open(data_url + "/expert_values_{}.csv".format(cluster), 'w') as f:
#     f.write("Point,X,Y,Mean,Std\n")
#     for p, x, y, mean, std in zip(dp, X, Y, location_mean, location_std):
#         f.write("%s,%s,%s,%s,%s\n" % (p, x, y, mean, std))