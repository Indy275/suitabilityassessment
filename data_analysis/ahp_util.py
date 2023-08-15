import pandas as pd
import numpy as np
import configparser
import os

config = configparser.ConfigParser()
parent = os.path.dirname
config.read(os.path.join(parent(parent(__file__)), 'config.ini'))
data_url = config['DEFAULT']['data_url']


def geo_mean(iterable):
    a = np.array(iterable)
    return (a+1).prod() ** (1.0 / len(a)) - 1


def read_dp_csv(cluster):
    df = pd.read_csv(data_url + f"/expert_point_info_{cluster[:2]}.csv")
    return df['Point'], df['Lng'], df['Lat']


def read_survey_data(name, date):
    df = pd.read_csv(data_url + '/{}.csv'.format(name))
    df = df.iloc[2:]
    df = df[pd.to_datetime(df['EndDate']) > pd.to_datetime(date)]
    df.columns = [c.replace(' ', '_') for c in df.columns]
    df.columns = [c.replace('.', '_') for c in df.columns]
    return df


def build_matrix(data):
    """ Create the PC matrix from the data
    :param data:
    :return: matrix of mx5x5 where m is the number of experts. 5 is hardcoded since every expert ranked 5 locations
    """
    n_experts = len(data)
    matrix = np.ones((n_experts, 5, 5))
    col_idx = [1, 2, 3, 4, 2, 3, 4, 3, 4, 4]
    row_idx = [0, 0, 0, 0, 1, 1, 1, 2, 2, 3]
    val_idx = [0, 7, 5, 3, 1, 3, 5, 7]  # AHP-like: with one option less (9x)
    s = 1.5  # scale parameter
    val_idx = [0, (1+s)**3, (1+s)**2, (1+s), 1, (1+s), (1+s)**2, (1+s)**3]  # geometric scale (Finan,Hurley): Transitive calibration of the AHP verbal scale

    for i, expert in enumerate(data.itertuples()):  # loop over expert comparison matrices
        for j, comp in enumerate(data.columns[-10:]):
            c = col_idx[j]
            r = row_idx[j]
            value = int(getattr(expert, comp))
            if value < 4:  # A more suitable
                matrix[i, r, c] = val_idx[value]
                matrix[i, c, r] = 1 / val_idx[value]
            elif value > 4:   # B more suitable
                matrix[i, r, c] = 1 / val_idx[value]
                matrix[i, c, r] = val_idx[value]
            else:  # value == 4 -> equally suitable
                matrix[i, r, c] = 1
                matrix[i, c, r] = 1
    return matrix


def compute_priority_weights(matrix):
    """
    Computes the priority weights of elements in a pairwise comparison matrix.
    First normalizes column-wise, then aggregates row-wise
    See e.g. https://www.spicelogic.com/docs/ahpsoftware/intro/ahp-calculation-methods-396 for explanation
    :param matrix: pairwise nxn comparison matrix
    :return: priority vector size n
    """
    return list(np.mean(matrix / np.sum(matrix, axis=0), axis=1).round(4))


def compute_consistency_ratio(matrix):
    n = len(matrix)
    lambda_max = np.max(np.linalg.eigvals(matrix))
    consistency_index = (lambda_max - n) / (n - 1)
    print("lambdamax , n",lambda_max, n)

    ri_dict = {3: 0.52, 4: 0.89, 5: 1.11, 6: 1.25, 7: 1.35, 8: 1.40, 9: 1.45,
               10: 1.49, 11: 1.52, 12: 1.54, 13: 1.56, 14: 1.58, 15: 1.59}

    random_index = ri_dict[n]
    consistency_ratio = np.real(consistency_index / random_index)
    return consistency_ratio
