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

numeric_data = 'Bodem en Watersturend OC_May 31, 2023_15.14'
cluster = 'OC'

data = ahp_util.read_survey_data(numeric_data, datetime.date(2023, 5, 15))
data.dropna(subset='Q2_1', inplace=True)  # Drop all reports in which weighting question was not answered
print("Number of participants that completed factorweighting:",len(data))
data.dropna(subset=data.columns[-10:], inplace=True)  # Drop all reports in which question was not answered
print("Number of participants that completed survey:",len(data))
duration = pd.to_numeric(data['Duration_(in_seconds)'])

matrix = ahp_util.build_matrix(data)
consistency_ratio = [ahp_util.compute_consistency_ratio(matrix[expert]).real for expert in range(len(data))]
priority_weights = [ahp_util.compute_priority_weights(matrix[expert]) for expert in range(len(data))]
print("Expert CR: Min, mean, and max consistency:", np.min(consistency_ratio),np.mean(consistency_ratio),np.max(consistency_ratio))
dura = np.array(duration)/60
dura = dura[dura<100]
print("Expert time taken: Min, mean, and max duration:", np.min(dura),np.mean(dura),np.max(dura))

location_gm = ahp_util.compute_priority_weights_aggregate(matrix)
print("Geometric mean: ", location_gm)

cr = ahp_util.compute_consistency_ratio(matrix.mean(axis=0))
location_am = ahp_util.compute_priority_weights(matrix.mean(axis=0))
print("Arithmetic mean: ", location_am, cr)

cr = ahp_util.compute_consistency_ratio(matrix.std(axis=0))
location_std = ahp_util.compute_priority_weights(matrix.std(axis=0))
print("Standard deviation:", location_std, cr)

mean_weight = ahp_util.compute_priority_weights_aggregate(matrix)
mean_std = ahp_util.compute_priority_weights(matrix.std(axis=0))

dp, X, Y = ahp_util.read_dp_csv(cluster)
point_data = pd.DataFrame({'Point': dp, 'Lng': X, 'Lat': Y})

weights = pd.DataFrame({'Point': list('ABCDE'), 'Value': mean_weight, 'Std': mean_std})
weights = weights.merge(point_data, on='Point', how='left')
weights = weights[['Point', 'Lng', 'Lat', 'Value', 'Std']]
weights.sort_values(by='Point', inplace=True)

weights.to_csv(data_url + "/expertscores_{}.csv".format(cluster), index=False)

