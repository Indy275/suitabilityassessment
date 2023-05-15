import os
from joblib import load
import configparser
from pathlib import Path

import pandas as pd
import rasterio
import numpy as np

config = configparser.ConfigParser()
parent = os.path.dirname
config.read(os.path.join(parent(parent(__file__)),'config.ini'))
data_url = config['DEFAULT']['data_url']


def load_xy(modifier, model=None):
    if model:
        df = pd.read_csv(data_url + '/' + modifier + '/' + model + ".csv")
        col_names = list(df.columns)
        df = df.to_numpy()

        X = df[:, :-1]
        y = df[:, -1]

        # load and reverse apply scaler
        ss = load(data_url + '/' + modifier + '/' + "scaler.joblib")
        df_orig = ss.inverse_transform(X)

        # prep labels for binary classification
        if model == 'hist_buildings':
            y[y < 0] = 0
            y[y > 0] = 1

        y = np.where(np.isnan(y), 0, y)

        return np.array(X), np.array(y), np.array(df_orig), col_names
    else:  # No labels are used, so take values of an arbitrary labeled df
        df = pd.read_csv(data_url + '/' + modifier + '/' + 'hist_buildings' + ".csv")
        col_names = list(df.columns)
        df = df.to_numpy()
        X = df[:, :-1]

        # load and reverse apply scaler
        ss = load(data_url + '/' + modifier +'/' + "scaler.joblib")
        df_orig = ss.inverse_transform(X)

        return np.array(X), np.array(df_orig), col_names


def load_bg(modifier):
    bg_tiff = data_url + '/' + modifier + '/bg_downscaled' + '.tif'
    if Path(bg_tiff).is_file():
        with rasterio.open(bg_tiff) as f:  # Write raster data to disk
            bg = f.read(1)
    else:
        bg = np.zeros((10, 10))
        bg += 1e-4
    return bg
