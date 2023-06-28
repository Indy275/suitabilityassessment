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

numeric_data = 'Bodem en Watersturend WS v2_June 9, 2023_09.44'
cluster = 'WS'

data = ahp_util.read_survey_data(numeric_data, datetime.date(2023, 5, 23))
data.dropna(subset='Q2_1', inplace=True)  # Drop all reports in which weighting question was not answered

weights_dp = []
datapoints = []
duration = []
for index, expert in data.iterrows():
    df_expert = data.iloc[[index - 2]]  # hardcoded -2 to account for header rows
    df_expert = df_expert.dropna(axis=1, how='all')
    if len(df_expert.columns) > 28:  # A completed survey has 29 columns: 10 comparisons; 6 other question; 13 metadata
        matrix = ahp_util.build_matrix(df_expert)
        points = list(df_expert.columns[-10].split('_')[1] + df_expert.columns[-9].split('_')[1][1] +
                      df_expert.columns[-1].split('_')[1])  # Gets the letters of the five locations from column names
        expertweights = ahp_util.compute_priority_weights(np.squeeze(matrix))
        print(expertweights)
        for point, exp_weight in zip(points, expertweights):
            weights_dp.append(exp_weight)
            datapoints.append(point)

        print("Expert evaluated:", points, "with these weights:", expertweights,
              "(CR: {})".format(ahp_util.compute_consistency_ratio(np.squeeze(matrix))))

dp, X, Y = ahp_util.read_dp_csv(cluster)
point_data = pd.DataFrame({'Point': dp, 'X': X, 'Y': Y})

print(datapoints, dp, X, Y)
weights = pd.DataFrame({'Point': datapoints, 'Value': weights_dp})
weights = weights.merge(point_data, on='Point', how='left')
weights = weights[['Point', 'X', 'Y', 'Value']]
weights.sort_values(by='Point', inplace=True)

weights.to_csv(data_url + "/expertscoresall_{}.csv".format(cluster), index=False)