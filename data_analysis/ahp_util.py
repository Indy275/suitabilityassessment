import datetime

import pandas as pd
import numpy as np
import configparser
import os

config = configparser.ConfigParser()
parent = os.path.dirname
config.read(os.path.join(parent(parent(__file__)), 'config.ini'))
data_url = config['DEFAULT']['data_url']


def build_matrix(data):
    n_experts = len(data)
    matrix = np.ones((n_experts, 5, 5))
    col_idx = [1, 2, 3, 4, 2, 3, 4, 3, 4, 4]
    row_idx = [0, 0, 0, 0, 1, 1, 1, 2, 2, 3]
    val_idx = [0, 7, 5, 3, 1, 3, 5, 7]
    for i, expert in enumerate(data.itertuples()):
        for j, comp in enumerate(data.columns[-10:]):
            c = col_idx[j]
            r = row_idx[j]
            value = int(getattr(expert, comp))
            if value < 4:  # A more suitable
                matrix[i, r, c] = val_idx[value]
                matrix[i, c, r] = 1 / val_idx[value]
            elif value > 4:
                matrix[i, r, c] = 1 / val_idx[value]
                matrix[i, c, r] = val_idx[value]
            else:  # value == 4 -> equally suitable
                matrix[i, r, c] = 1
                matrix[i, c, r] = 1
    return matrix


def compute_consistency_ratio(matrix):
    n = len(matrix)
    # Calculate the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    max_eigenvalue_index = np.argmax(eigenvalues)
    eigenvector = eigenvectors[:, max_eigenvalue_index]

    weighted_sum = np.dot(matrix, eigenvector)
    max_eigenvalue = np.sum(weighted_sum / eigenvector) / n

    consistency_index = (max_eigenvalue - n) / (n - 1)

    ri_dict = {3: 0.52, 4: 0.89, 5: 1.11, 6: 1.25, 7: 1.35, 8: 1.40, 9: 1.45,
               10: 1.49, 11: 1.52, 12: 1.54, 13: 1.56, 14: 1.58, 15: 1.59}

    random_index = ri_dict[n]
    consistency_ratio = consistency_index / random_index


    return consistency_ratio


def compute_priority_weights(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    max_eigenvalue_index = np.argmax(eigenvalues)

    priority_vector = np.real(eigenvectors[:, max_eigenvalue_index])

    # Normalize the priority vector
    priority_weights = priority_vector / np.sum(priority_vector) * 10
    return list(priority_weights)


def read_dp_csv(cluster):
    df = pd.read_csv(data_url + "/expert_points_{}.csv".format(cluster))
    return df['point'], df['X'], df['Y']


def read_survey_data(name, date):
    df = pd.read_csv(data_url + '/collected_data/{}.csv'.format(name))
    df = df.iloc[2:]
    df = df[pd.to_datetime(df['EndDate']) > pd.to_datetime(date)]
    df.columns = [c.replace(' ', '_') for c in df.columns]
    return df

