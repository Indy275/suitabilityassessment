import os
import csv
from joblib import load
import itertools
import configparser
from pathlib import Path
import matplotlib.image as mpimg
import pandas as pd
import numpy as np

config = configparser.ConfigParser()
parent = os.path.dirname
config.read(os.path.join(parent(parent(__file__)), 'config.ini'))
data_url = config['DEFAULT']['data_url']


def load_xy(modifier, model=None):
    if not model:  # no model means labels are not used; open expert_ref arbitrarily as labels are dropped anyway
        ref_std = 'expert_ref'
    else:
        ref_std = model

    df = pd.read_csv(data_url + '/' + modifier + '/' + ref_std + ".csv")
    col_names = list(df.columns)
    df = df.to_numpy()
    X = df[:, :-1]

    # load and reverse apply scaler
    ss = load(data_url + '/' + modifier + '/' + "scaler.joblib")
    df_orig = ss.inverse_transform(X[:, 2:])

    if not model:
        return np.array(X), np.array(df_orig), col_names

    y = df[:, -1]
    if model == 'hist_buildings':  # Prepare label binarization for historic buildings
        y[y < 0] = 0
        y[y > 0] = 1

    y = np.where(np.isnan(y), 0, y)  # No house is built on NaN cells

    return np.array(X), np.array(y), np.array(df_orig), col_names


def load_expert(modifier):
    X, y = [], []

    def flatten(l):
        l2 = [([x] if isinstance(x, str) else x) for x in l]
        return list(itertools.chain(*l2))

    with open(data_url + '/expertscores_{}.csv'.format(modifier), 'r') as scores_f, open(
            data_url + '/expert_point_info_{}.csv'.format(modifier), 'r') as info_f:
        scores_reader = csv.reader(scores_f)
        info_reader = csv.reader(info_f)
        next(scores_reader)
        col_names = next(info_reader)
        for s_row, i_row in zip(scores_reader, info_reader):
            X.append(flatten([s_row[1], s_row[2], i_row[1:6]]))  # X1, X2, features
            y.append(s_row[3])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), col_names[1:6]


def load_expert_all(modifier):
    all_exp_scores = pd.read_csv(data_url + '/expertscoresall_{}.csv'.format(modifier), header=[0])
    exp_point_info = pd.read_csv(data_url + '/expert_point_info_{}.csv'.format(modifier), header=[0])
    point_info_scores = all_exp_scores.merge(exp_point_info, on='Point', how='left',
                                             suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
    print(point_info_scores.columns)
    y = point_info_scores['Value']
    point_info_scores.drop(['Value', 'Point', 'Unnamed: 0'], axis=1, inplace=True)
    X = point_info_scores
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), point_info_scores.columns[2:]


# def transform_expert(mod):
#     if mod[-3:] == 'all':
#         X, y, col_names = load_expert_all(mod)
#     else:
#         X, y, col_names = load_expert(mod)
#
#     X_feats = X[:, 2:7]
#     X_loc_feats = X
#
#     return X, X_loc_feats, y, col_names


def load_data(mod, ref_std=None):
    if ref_std == 'expert_ref' and mod.lower() in ['oc', 'ocall', 'ws', 'wsall']:  # Model is expert ref and test labels are available
        if mod[-3:] == 'all':
            X, y, col_names = load_expert_all(mod)
        else:
            X, y, col_names = load_expert(mod)
        X_feats = X[:, 2:7]
        X_loc_feats = X
        return X_feats, X_loc_feats, y, col_names
    elif ref_std == 'hist_buildings' and mod.lower() in ['purmer', 'schermerbeemster', 'purmerend', 'volendam']:  # Model is hist_buildings and test labels are available
        return load_xy(mod, model=ref_std)
    else:  # No labels are used, so arbitrarily take expert ref data since labels are dropped anyway
        return load_xy(mod)


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