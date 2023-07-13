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


def set_cluster(cluster):
    if cluster == 'WS':
        data_link = 'Bodem en Watersturend WS v2_June 9, 2023_09.44'
        start_date = datetime.date(2023, 5, 23)
        locationIDs = list('ABCDEFGHIJKLMNOPQRSTUVWXYZab')
    else:  # cluster == 'OC':
        data_link = 'Bodem en Watersturend OC_May 31, 2023_15.14'
        start_date = datetime.date(2023, 5, 15)
        locationIDs = list('ABCDE')
    return data_link, start_date, locationIDs


def process_experts_indiv(data, locationIDs):
    weights = pd.DataFrame(columns=locationIDs)
    n_completed_surveys = 0
    for index, expert in data.iterrows():
        df_expert = expert.to_frame().T.dropna(axis=1, how='all')
        if len(df_expert.columns) > 28:  # A completed survey has 29 columns: 10 comparisons; 6 other question; 13 metadata
            n_completed_surveys += 1
            matrix = ahp_util.build_matrix(df_expert)
            points = [df_expert.columns[-10].split('_')[1] + df_expert.columns[-9].split('_')[1][1] +
                          df_expert.columns[-1].split('_')[1]]
            weight_dict = {point: 0 for point in locationIDs}
            expertweights = ahp_util.compute_priority_weights(np.squeeze(matrix))
            for point, exp_weight in zip(points, expertweights):
                weight_dict[point] = exp_weight
            print("Expert evaluated:", points, "with these weights:", expertweights,
                  "(CR: {})".format(ahp_util.compute_consistency_ratio(np.squeeze(matrix))))
            new_df = pd.DataFrame(weight_dict, index=[0])
            weights = pd.concat([weights, new_df], ignore_index=True)
    return weights


def process_experts_group(data):
    datapoints, weights_dp = [], []
    for index, expert in data.iterrows():
        df_expert = expert.to_frame().T.dropna(axis=1, how='all')
        if len(df_expert.columns) > 28:  # A completed survey has 29 columns: 10 comparisons; 6 other question; 13 metadata
            matrix = ahp_util.build_matrix(df_expert)
            points = [df_expert.columns[-10].split('_')[1] + df_expert.columns[-9].split('_')[1][1] +
                          df_expert.columns[-1].split('_')[1]]  # Gets the letters of the five locations from column names
            expertweights = ahp_util.compute_priority_weights(np.squeeze(matrix))
            for point, exp_weight in zip(points, expertweights):
                weights_dp.append(exp_weight)
                datapoints.append(point)

            print("Expert evaluated:", points, "with these weights:", expertweights,
                  "(CR: {})".format(ahp_util.compute_consistency_ratio(np.squeeze(matrix))))
    return datapoints, weights_dp


def run_model(cluster, expert_processing):
    data_link, start_date, locationIDs = set_cluster(cluster)

    dp, X, Y = ahp_util.read_dp_csv(cluster)
    point_data = pd.DataFrame({'Point': dp, 'Lng': X, 'Lat': Y})

    data = ahp_util.read_survey_data(data_link, start_date)
    data.dropna(subset='Q2_1', inplace=True)  # Drop all reports in which weighting question was not answered
    print("Number of participants that completed factorweighting:", len(data))

    if expert_processing.startswith('i'):  # individual
        weights = process_experts_indiv(data, locationIDs)
        mean_weight = weights.replace(0, np.NaN).mean(axis=0).replace(np.NaN, 0)
        mean_std = weights.replace(0, np.NaN).std(axis=0).replace(np.NaN, 0)
        weights = pd.DataFrame({'Point': locationIDs, 'Value': mean_weight, 'Std': mean_std})

    elif expert_processing.startswith('g'):  # group
        datapoints, weights = process_experts_group(data)
        weights = pd.DataFrame({'Point': datapoints, 'Value': weights})

    weights = weights.merge(point_data, on='Point', how='left')
    # weights = weights[['Point', 'Lng', 'Lat', 'Value', 'Std']]
    weights.sort_values(by='Point', inplace=True)

    weights.to_csv(data_url + "/expertscores_{}.csv".format(cluster), index=False)
