import datetime

import configparser
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_analysis import ahp_util

config = configparser.ConfigParser()
parent = os.path.dirname
config.read(os.path.join(parent(parent(__file__)), 'config.ini'))
data_url = config['DEFAULT']['data_url']

choice_data = 'Bodem en Watersturend OC_May 17, 2023_11.51'
numeric_data = 'Bodem en Watersturend OC_May 31, 2023_15.14'
cluster = 'OC'

df = ahp_util.read_survey_data(choice_data, datetime.date(2023, 5, 15))
data = ahp_util.read_survey_data(numeric_data, datetime.date(2023, 5, 15))
data.dropna(subset='Q2_1', inplace=True)  # Drop all reports in which weighting question was not answered
print("Number of participants that completed factorweighting:",len(data))
data.dropna(subset=data.columns[-10:], inplace=True)  # Drop all reports in which question was not answered
print("Number of participants that completed survey:",len(data))
duration = pd.to_numeric(data['Duration_(in_seconds)'])

matrix = ahp_util.build_matrix(data)
consistency_ratio = [ahp_util.compute_consistency_ratio(matrix[expert]) for expert in range(len(data))]
priority_weights = [ahp_util.compute_priority_weights(matrix[expert]) for expert in range(len(data))]
print("mean per location ", np.array(priority_weights).mean(axis=0))
print("std per location ", np.array(priority_weights).std(axis=0))
print("weights per expert:",priority_weights)
print("consistency ratios:", consistency_ratio)
print("min mean max consistency:", np.min(consistency_ratio),np.mean(consistency_ratio),np.max(consistency_ratio))
dura = np.array(duration)/60
print(np.array(duration)/60)
dura = dura[dura<100]
print("min mean max duration:", np.min(dura),np.mean(dura),np.max(dura))

location_gm = ahp_util.compute_priority_weights_aggregate(matrix)
print("Geometric mean: ", location_gm)

cr = ahp_util.compute_consistency_ratio(matrix.mean(axis=0))
location_am = ahp_util.compute_priority_weights(matrix.mean(axis=0))
print("Arithmetic mean: ", location_am, cr)

cr = ahp_util.compute_consistency_ratio(matrix.std(axis=0))
location_std = ahp_util.compute_priority_weights(matrix.std(axis=0))
print("Standard deviation:", location_std, cr)

dp, X, Y = ahp_util.read_dp_csv(cluster)

# with open(data_url + "/expertscores_{}.csv".format(cluster), 'w') as f:
#     f.write("Point,X,Y,Mean,Std\n")
#     for p, x, y, mean, std in zip(dp, X, Y, location_gm, location_std):
#         f.write("%s,%s,%s,%s,%s\n" % (p, x, y, mean, std))

ratio2 = [(0.41913123688450415+0j), (0.2814425675943259+0j), (0.5795550347978271+0j), (0.1447406877493689+0j), (0.6756547403936265+0j), (0.14944429913930798+0j), (0.2809649680632276+0j), (0.24101441674116333+0j), (0.19959397671736384+0j), (0.21996983966116834+0j), (0.5079029065848542+0j), (0.12372896167266693+0j)]
plt.boxplot([ratio2, consistency_ratio])
labels = ['WS', 'OC']
plt.xticks(ticks=range(1, len(labels) + 1), labels=labels)
plt.ylim([0,1])
left, right = plt.xlim()
plt.hlines(y=0.1, xmin=left, xmax=right, linestyles='dashed', colors=['red'])
plt.show()