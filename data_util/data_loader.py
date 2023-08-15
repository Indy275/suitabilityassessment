import os
from joblib import load
import configparser
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

from data_util import create_data

config = configparser.ConfigParser()
parent = os.path.dirname
config.read(os.path.join(parent(parent(__file__)), 'config.ini'))
data_url = config['DEFAULT']['data_url']
coords_as_features = config.getboolean('DATA_SETTINGS', 'coords_as_features')


class DataLoader:
    def __init__(self, modifier, ref_std):
        self.modifier = modifier
        self.ref_std = ref_std  # hist_buildings, expert_ref or testdata
        self.X = None  # (x, y, features)
        self.lnglat = None  # (x, y)
        self.col_names = None
        self.y = None
        self.nans = None
        self.load_data()
        self.bbox, self.size = load_meta(self.modifier)
        self.X_orig = self.load_orig_df()

    def load_orig_df(self):
        ss = load(data_url + '/' + self.modifier + "/" + self.ref_std + "_scaler.joblib")
        df_orig = ss.inverse_transform(self.X)
        self.lnglat = df_orig[:, :2]
        return df_orig

    # def denormalize(self, data):
    #     ss = load(data_url + '/' + self.modifier + "/" + self.ref_std + "_scaler.joblib")
    #     print(data.shape, self.X_orig.shape)
    #     if data.shape[-1] != self.X_orig.shape[-1]:  # data misses the coordinates, add coordinate data
    #         data = np.column_stack((self.lnglat, self.X))
    #     print(data.shape, self.X_orig.shape)
    #
    #     data = self.X_orig[~self.nans]
    #
    #     data = ss.inverse_transform(data)
    #     data = data[~self.nans]
    #
    #     return data

    def load_data(self):
        df = pd.read_csv(data_url + '/' + self.modifier + '/' + self.ref_std + '.csv')
        col_names = list(df.columns)
        self.X = np.array(df.to_numpy()[:, :-1])
        y = df.to_numpy()[:, -1]
        if self.ref_std == 'hist_buildings':
            y = np.where(y < 0, 0, np.where(y > 0, 1, np.where(np.isnan(y), 0, y)))
        self.y = np.array(y)
        self.col_names = col_names[:-1]

    def _preprocess_test(self):
        if coords_as_features:
            X = self.X
            col_names = self.col_names
        else:
            X = self.X[:, 2:]
            col_names = self.col_names[2:]
        self.nans = np.isnan(self.X).any(axis=1)
        return X, self.nans, self.lnglat, self.size, col_names

    def _preprocess_train(self):
        if coords_as_features:
            X = self.X[self.y != 0]
            col_names = self.col_names
        else:
            X = self.X[:, 2:]
            X = X[self.y != 0]
            col_names = self.col_names[2:]
        # self.nans = np.isnan(self.X).any(axis=1)
        y = self.y[self.y != 0]
        return X, y, self.lnglat, col_names

    def preprocess_input(self):
        if self.ref_std == 'testdata':
            return self._preprocess_test()
        elif self.ref_std in ['hist_buildings', 'expert_ref']:
            return self._preprocess_train()

    def load_bg(self):
        bg_png = data_url + '/' + self.modifier + '/bg.tif'
        if not Path(bg_png).is_file():
            create_data.create_bg(self.modifier, self.bbox)
        bg = Image.open(bg_png)
        return bg


def load_meta(modifier):
    df = pd.read_csv(data_url + '/metadata.csv', delimiter=';', index_col='modifier')
    try:
        row = df.loc[modifier]
    except Exception:
        print(f"Error: No metadata found for '{modifier}' ")
        return

    bbox = row['bbox']
    bbox = np.sort([float(i.strip()[0:-1]) for i in bbox.split(',')])
    bbox_ordered = [bbox[0], bbox[2], bbox[1], bbox[3]]
    if not np.isnan(row['size']):
        size = int(row['size'])
    else:
        size = 400
    return bbox_ordered, size
