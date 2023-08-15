import datetime
import configparser
import os
import numpy as np
import pandas as pd
# from scipy.stats import gmean
import scipy.stats as stats

from data_analysis import ahp_util

config = configparser.ConfigParser()
parent = os.path.dirname
config.read(os.path.join(parent(parent(__file__)), 'config.ini'))
data_url = config['DEFAULT']['data_url']


def set_cluster(cluster):
    if cluster == 'ws':
        data_link = 'Bodem en Watersturend WS v2_August 3, 2023_14.01'
        start_date = datetime.date(2023, 5, 23)
        locationIDs = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcd')
    else:  # cluster == 'OC':
        data_link = 'Bodem en Watersturend OC_May 31, 2023_15.14'
        start_date = datetime.date(2023, 5, 15)
        locationIDs = list('ABCDE')
    return data_link, start_date, locationIDs


def process_experts_indiv(data):
    datapoints, weights_dp = [], []
    crs = []
    for index, expert in data.iterrows():
        df_expert = expert.to_frame().T.dropna(axis=1, how='all')
        if len(df_expert.columns) > 28:  # A completed survey has 29 columns: 10 comparisons; 6 other question; 13 metadata
            matrix = ahp_util.build_matrix(df_expert)
            points = list(df_expert.columns[-10].split('_')[1] + df_expert.columns[-9].split('_')[1][1] +
                          df_expert.columns[-1].split('_')[1])  # Gets the letters of the five locations from column names
            expertweights = ahp_util.compute_priority_weights(np.squeeze(matrix))

            for point, exp_weight in zip(points, expertweights):
                weights_dp.append(exp_weight)
                datapoints.append(point)
            cr = ahp_util.compute_consistency_ratio(np.squeeze(matrix))
            crs.append(cr)
            print(f'Expert evaluated {points} with weights: {expertweights} (CR: {cr})')
    print("consistencyratios",crs)
    return datapoints, weights_dp


def run_model(cluster, expert_processing):
    data_link, start_date, locationIDs = set_cluster(cluster)

    dp, X, Y = ahp_util.read_dp_csv(cluster)
    point_data = pd.DataFrame({'Point': dp, 'Lng': X, 'Lat': Y})

    data = ahp_util.read_survey_data(data_link, start_date)
    data.dropna(subset='Q2_1', inplace=True)  # Drop all reports in which weighting question was not answered
    print("Number of participants that completed factorweighting:", len(data))

    if expert_processing.startswith('g'):  # group
        datapoints, weights_raw = process_experts_indiv(data)
        raw_weights = pd.DataFrame({'Point': datapoints, 'Value': weights_raw})
        weights = raw_weights.groupby(raw_weights.Point).apply(stats.gmean).to_frame().reset_index()
        weights.rename(columns={0: 'Value'}, inplace=True)
        weights['Value'] = weights['Value'].astype(float)

    elif expert_processing.startswith('i'):  # individual
        cluster = cluster + 'all'
        datapoints, weights = process_experts_indiv(data)
        weights = pd.DataFrame({'Point': datapoints, 'Value': weights})

    weights = weights.merge(point_data, on='Point', how='left')
    weights.sort_values(by='Point', inplace=True)

    weights.to_csv(data_url + "/expertscores_{}.csv".format(cluster), index=False)
