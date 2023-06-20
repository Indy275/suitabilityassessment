import os
import csv
from joblib import load
import itertools
import configparser
from pathlib import Path
import matplotlib.image as mpimg

import pandas as pd
import rasterio
import numpy as np

config = configparser.ConfigParser()
parent = os.path.dirname
config.read(os.path.join(parent(parent(__file__)), 'config.ini'))
data_url = config['DEFAULT']['data_url']


def load_xy(modifier, model=None):
    if model:
        df = pd.read_csv(data_url + '/' + modifier + '/' + model + ".csv")
        # df.drop(['Overstromingsbeeld primaire keringen', 'Overstromingsbeeld regionale keringen', 'Bodemdaling Huidig', 'inundatiediepte T100'], axis=1, inplace=True)
        col_names = list(df.columns)
        df = df.to_numpy()
        X = df[:, :-1]
        y = df[:, -1]

        # load and reverse apply scaler
        ss = load(data_url + '/' + modifier + '/' + "scaler.joblib")
        df_orig = ss.inverse_transform(X[:, 2:])
        # prep labels for binary classification
        if model == 'hist_buildings':
            y[y < 0] = 0
            y[y > 0] = 1

        y = np.where(np.isnan(y), 0, y)

        return np.array(X), np.array(y), np.array(df_orig), col_names
    else:  # No labels are used, so take values of an arbitrary labeled df
        # df = pd.read_csv(data_url + '/' + modifier + '/' + 'hist_buildings' + ".csv")
        df = pd.read_csv(data_url + '/' + modifier + '/' + 'expert_ref' + ".csv")

        col_names = list(df.columns)
        df = df.to_numpy()
        X = df[:, :-1]

        # load and reverse apply scaler
        ss = load(data_url + '/' + modifier + '/' + "scaler.joblib")
        df_orig = ss.inverse_transform(X[:, 2:])

        return np.array(X), np.array(df_orig), col_names


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


def flatten(l):
    l2 = [([x] if isinstance(x, str) else x) for x in l]
    return list(itertools.chain(*l2))


def load_expert(modifier):
    X, y = [], []
    with open(data_url + '/expertscores_{}.csv'.format(modifier), 'r') as scores_f, open(
            data_url + '/expert_point_info_{}.csv'.format(modifier), 'r') as info_f:
        scores_reader = csv.reader(scores_f)
        info_reader = csv.reader(info_f)
        next(scores_reader)
        next(info_reader)
        for s_row, i_row in zip(scores_reader, info_reader):
            X.append(flatten([s_row[1], s_row[2], i_row[1:6]]))  # X1, X2, features
            y.append(s_row[3])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
