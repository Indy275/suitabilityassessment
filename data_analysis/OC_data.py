import datetime

import configparser
import os
from data_analysis import ahp_util

config = configparser.ConfigParser()
parent = os.path.dirname
config.read(os.path.join(parent(parent(__file__)), 'config.ini'))
data_url = config['DEFAULT']['data_url']

choice_data = 'Bodem en Watersturend OC_May 17, 2023_11.51'
numeric_data = 'Bodem en Watersturend OC_May 30, 2023_13.29'
cluster = 'OC'

df = ahp_util.read_survey_data(choice_data, datetime.date(2023, 5, 15))
data = ahp_util.read_survey_data(numeric_data, datetime.date(2023, 5, 15))
data.dropna(subset=data.columns[-10:], inplace=True)  # Drop all reports in which question was not answered
matrix = ahp_util.build_matrix(data)
consistency_ratio = [ahp_util.compute_consistency_ratio(matrix[expert]) for expert in range(len(data))]
priority_weights = [ahp_util.compute_priority_weights(matrix[expert]) for expert in range(len(data))]
print("weights per expert:",priority_weights, consistency_ratio)

location_gm = ahp_util.compute_priority_weights_aggregate(matrix)
print("Geometric mean: ", location_gm)

cr = ahp_util.compute_consistency_ratio(matrix.mean(axis=0))
location_am = ahp_util.compute_priority_weights(matrix.mean(axis=0))
print("Arithmetic mean: ", location_am, cr)

cr = ahp_util.compute_consistency_ratio(matrix.std(axis=0))
location_std = ahp_util.compute_priority_weights(matrix.std(axis=0))
print("Standard deviation:", location_std, cr)

dp, X, Y = ahp_util.read_dp_csv(cluster)

with open(data_url + "/expertscores_{}.csv".format(cluster), 'w') as f:
    f.write("Point,X,Y,Mean,Std\n")
    for p, x, y, mean, std in zip(dp, X, Y, location_gm, location_std):
        f.write("%s,%s,%s,%s,%s\n" % (p, x, y, mean, std))

# for i in range(len(data)):
#     priority_weights = ahp_util.compute_priority_weights(matrix[i])
#     with open(data_url + "/expertscores_{}{}.csv".format(cluster, i), 'w') as f:
#         f.write("Point,X,Y,val\n")
#         for p, x, y, mean, std in zip(dp, X, Y, priority_weights, location_std):
#             f.write("%s,%s,%s,%s\n" % (p, x, y, mean[0]))
