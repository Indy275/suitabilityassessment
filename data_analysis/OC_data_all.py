import datetime

import configparser
import os
import pandas as pd
from itertools import cycle

from data_analysis import ahp_util

config = configparser.ConfigParser()
parent = os.path.dirname
config.read(os.path.join(parent(parent(__file__)), 'config.ini'))
data_url = config['DEFAULT']['data_url']

numeric_data = 'Bodem en Watersturend OC_May 31, 2023_15.14'
cluster = 'OC'

data = ahp_util.read_survey_data(numeric_data, datetime.date(2023, 5, 15))
data.dropna(subset='Q2_1', inplace=True)  # Drop all reports in which weighting question was not answered
print("Number of participants that completed factorweighting:",len(data))
data.dropna(subset=data.columns[-10:], inplace=True)  # Drop all reports in which question was not answered
print("Number of participants that completed survey:",len(data))
duration = pd.to_numeric(data['Duration_(in_seconds)'])

weights_dp = []
datapoints = []
matrix = ahp_util.build_matrix(data)
expertweights = [ahp_util.compute_priority_weights(matrix[expert]) for expert in range(len(data))]
print(expertweights)
points = list('ABCDE')
for expert in expertweights:
    for point, exp_weight in zip(cycle(points), expert):
        weights_dp.append(exp_weight)
        datapoints.append(point)

dp, X, Y = ahp_util.read_dp_csv(cluster)
point_data = pd.DataFrame({'Point': dp, 'X': X, 'Y': Y})

print(datapoints, dp, X, Y)
weights = pd.DataFrame({'Point': datapoints, 'Value': weights_dp})
weights = weights.merge(point_data, on='Point', how='left')
weights = weights[['Point', 'X', 'Y', 'Value']]
weights.sort_values(by='Point', inplace=True)

weights.to_csv(data_url + "/expertscoresall_{}.csv".format(cluster), index=False)