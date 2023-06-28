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
print("Number of participants that completed factorweighting:",len(data))

weights = pd.DataFrame(columns=list('ABCDEFGHIJKLMNOPQRSTUVWXYZab'))
n_completed_surveys = 0
consistency_ratio = []
duration = []
for index, expert in data.iterrows():
    df_expert = data.iloc[[index-2]]  # -2 is hardcoded since index 2 is first row (before is header)
    df_expert = df_expert.dropna(axis=1, how='all')
    if len(df_expert.columns) > 28:  # A completed survey has 29 columns: 10 comparisons; 6 other question; 13 metadata
        print("expert in",df_expert['Q1'])
        n_completed_surveys +=1
        matrix = ahp_util.build_matrix(df_expert)
        points = list(df_expert.columns[-10].split('_')[1] + df_expert.columns[-9].split('_')[1][1] + df_expert.columns[-1].split('_')[1])
        weight_dict = dict.fromkeys(list('ABCDEFGHIJKLMNOPQRSTUVWXYZab'), 0)
        expertweights = ahp_util.compute_priority_weights(np.squeeze(matrix))
        for point, exp_weight in zip(points, expertweights):
            weight_dict[point] = exp_weight
        print("Expert evaluated:", points, "with these weights:", expertweights, "(CR: {})".format(ahp_util.compute_consistency_ratio(np.squeeze(matrix))))
        new_df = pd.DataFrame(weight_dict, index=[0])
        weights = pd.concat([weights, new_df], ignore_index=True)
        consistency_ratio.append(ahp_util.compute_consistency_ratio(np.squeeze(matrix)))
        if int(df_expert['Duration_(in_seconds)']) < 3600:
            duration.append(int(df_expert['Duration_(in_seconds)']))
        else:
            print("expert took very long:",int(df_expert['Duration_(in_seconds)']))

print("Number of participants that completed survey:",n_completed_surveys)
print("min mean max consistency:", np.min(consistency_ratio),np.mean(consistency_ratio),np.max(consistency_ratio))
print("constratio", consistency_ratio)
print(np.array(duration)/60)
dura = np.array(duration)/60
dura = dura[dura<100]
print("min mean max duration:", np.min(dura),np.mean(dura),np.max(dura))

mean_weight = weights.replace(0, np.NaN).mean(axis=0).replace(np.NaN,0)
# TODO: replace with geometric mean to follow Saaty's AHP
mean_std = weights.replace(0, np.NaN).std(axis=0).replace(np.NaN,0)

dp, X, Y = ahp_util.read_dp_csv(cluster)

with open(data_url + "/expertscores_{}.csv".format(cluster), 'w') as f:
    f.write("Point,X,Y,Mean,Std\n")
    for p, x, y, mean, std in zip(dp, X, Y, mean_weight, mean_std):
        f.write("%s,%s,%s,%s,%s\n" % (p, x, y, mean, std))
