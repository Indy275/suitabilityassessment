import os
from joblib import load
import configparser
from pathlib import Path
import matplotlib.image as mpimg
import pandas as pd
import numpy as np

config = configparser.ConfigParser()
parent = os.path.dirname
config.read(os.path.join(parent(parent(__file__)), 'config.ini'))
data_url = config['DEFAULT']['data_url']


def load_orig_df(mod):
    df = pd.read_csv(data_url + '/' + mod + '/testdata.csv')
    col_names = list(df.columns)
    df = df.to_numpy()
    X = df[:, :-1]

    # load and reverse apply scaler
    ss = load(data_url + '/' + mod + '/' + "scaler.joblib")
    df_orig = ss.inverse_transform(X[:, 2:])

    return df_orig, col_names


# def load_histbld(modifier, model):
#     df = pd.read_csv(data_url + '/' + modifier + '/' + model + '.csv')
#     col_names = list(df.columns)
#     df = df.to_numpy()
#     X = df[:, :-1]
#
#     y = df[:, -1]
#
#     if model == 'hist_buildings':
#         y[y < 0] = 0
#         y[y > 0] = 1
#     y = np.where(np.isnan(y), 0, y)  # No house is built on NaN cells
#
#     return np.array(X), np.array(y), col_names[:-1]
#
#
# def load_expert(modifier, all):
#     df = pd.read_csv(data_url + '/' + modifier + "/expert_ref.csv", header=[0])
#     col_names = list(df.columns)
#
#     if all:
#         all_exp_scores = pd.read_csv(data_url + f'/expertscoresall_{modifier}.csv', header=[0])
#     else:
#         all_exp_scores = pd.read_csv(data_url + f'/expertscores_{modifier}.csv', header=[0])
#     exp_point_info = pd.read_csv(data_url + '/expert_point_info_{}.csv'.format(modifier), header=[0])
#     point_info_scores = all_exp_scores.merge(exp_point_info, on='Point', how='left',
#                                              suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
#     y = point_info_scores['Value']
#     point_info_scores.drop(['Value', 'Std', 'Point', 'Unnamed: 0'], axis=1, inplace=True, errors='ignore')
#     X = point_info_scores
#     col_names = list(point_info_scores.columns)
#
#     return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), col_names


# def load_expert_all(modifier):
#     all_exp_scores = pd.read_csv(data_url + '/expertscoresall_{}.csv'.format(modifier), header=[0])
#     exp_point_info = pd.read_csv(data_url + '/expert_point_info_{}.csv'.format(modifier), header=[0])
#     point_info_scores = all_exp_scores.merge(exp_point_info, on='Point', how='left',
#                                              suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
#     print(point_info_scores.columns)
#     y = point_info_scores['Value']
#     point_info_scores.drop(['Value', 'Point', 'Unnamed: 0'], axis=1, inplace=True)
#     X = point_info_scores
#     return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), point_info_scores.columns[2:]


def load_xy(mod, model=None):
    df = pd.read_csv(data_url + '/' + mod + '/' + model + '.csv')
    col_names = list(df.columns)
    df = df.to_numpy()
    X = df[:, :-1]

    y = df[:, -1]

    if model == 'hist_buildings':
        y[y < 0] = 0
        y[y > 0] = 1
    y = np.where(np.isnan(y), 0, y)  # No house is built on NaN cells

    return np.array(X), np.array(y), col_names[:-1]


def load_x(modifier):
    df = pd.read_csv(data_url + '/' + modifier + '/testdata.csv')
    col_names = list(df.columns)
    X = df.to_numpy()[:, :-1]
    return np.array(X), col_names[:-1]


def load_bg(modifier):
    bg_png = data_url + '/' + modifier + '/bg_downscaled' + '.png'
    if Path(bg_png).is_file():
        # with rasterio.open(bg_tiff) as f:  # Write raster data to disk
        #     bg = f.read(1)
        bg = mpimg.imread(bg_png)
    else:
        bg = np.zeros((10, 10))
        bg += 1e-4
    return bg