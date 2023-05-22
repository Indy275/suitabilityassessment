import datetime

import configparser
import os
from data_analysis import ahp_util

config = configparser.ConfigParser()
parent = os.path.dirname
config.read(os.path.join(parent(parent(__file__)), 'config.ini'))
data_url = config['DEFAULT']['data_url']

numeric_data = ''
choice_data = ''
cluster = 'WS'

df = ahp_util.read_survey_data(choice_data, datetime.date(2023, 5, 23))
data = ahp_util.read_survey_data(numeric_data, datetime.date(2023, 5, 23))

matrix = ahp_util.build_matrix(data)
consistency_ratio = [ahp_util.compute_consistency_ratio(matrix[expert]) for expert in range(len(data))]
priority_weights = [ahp_util.compute_priority_weights(matrix[expert]) for expert in range(len(data))]

cr = ahp_util.compute_consistency_ratio(matrix.mean(axis=0))
location_mean = ahp_util.compute_priority_weights(matrix.mean(axis=0))
print(location_mean, cr)

cr = ahp_util.compute_consistency_ratio(matrix.std(axis=0))
location_std = ahp_util.compute_priority_weights(matrix.std(axis=0))
print(location_std, cr)

dp, X, Y = ahp_util.read_dp_csv(cluster)

with open(data_url + "/expert_values_{}.csv".format(cluster), 'w') as f:
    f.write("Point,X,Y,Mean,Std\n")
    for p, x, y, mean, std in zip(dp, X, Y, location_mean, location_std):
        f.write("%s,%s,%s,%s,%s\n" % (p, x, y, mean, std))