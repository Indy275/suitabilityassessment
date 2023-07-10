import os
from joblib import load
import configparser
from pathlib import Path
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
from PIL import Image

config = configparser.ConfigParser()
parent = os.path.dirname
config.read(os.path.join(parent(parent(__file__)), 'config.ini'))
data_url = config['DEFAULT']['data_url']
coords_as_features = config.getboolean('DATA_SETTINGS','coords_as_features')


class DataLoader:
    def __init__(self, modifier, model):
        self.modifier = modifier
        self.model = model  # hist_buildings, expert_ref or testdata
        self.X = None
        self.lnglat = None
        self.col_names = None
        self.y = None
        self.load_data()

    def load_orig_df(self):
        # load and reverse apply scaler
        ss = load(data_url + '/' + self.modifier + "/" + self.model + "_scaler.joblib")
        df_orig = ss.inverse_transform(self.X)
        return df_orig

    def load_data(self):
        df = pd.read_csv(data_url + '/' + self.modifier + '/' + self.model + '.csv')
        col_names = list(df.columns)
        X_lnglat_feat = df.to_numpy()[:, :-1]
        self.X = np.array(X_lnglat_feat[:, 2:])
        self.lnglat = np.array(X_lnglat_feat[:, :2])
        y = df[:, -1]
        if self.model == 'hist_buildings':
            y[y < 0] = 0
            y[y > 0] = 1
        y = np.where(np.isnan(y), 0, y)  # No house is built on NaN cells
        self.y = np.array(y)
        self.col_names = col_names[:-1]

    def _preprocess_test(self):
        if coords_as_features:
            X = self.X
            col_names = self.col_names
        else:
            X = self.X[:, 2:]
            col_names = self.col_names[2:]
        nans = np.isnan(self.X).any(axis=1)
        X = X[~nans]
        return X, nans, self.lnglat, col_names

    def _preprocess_train(self):
        if coords_as_features:
            X = self.X[self.y != 0]
            col_names = self.col_names
        else:
            X = self.X[:, 2:]
            X = X[self.y != 0]
            col_names = self.col_names[2:]
        y = self.y[self.y != 0]
        return X, y, self.lnglat, col_names

    def preprocess_input(self):
        if self.model == 'testdata':
            return self._preprocess_test()
        elif self.model in ['hist_buildings', 'expert_ref']:
            return self._preprocess_train()

    def load_bg_png(self):
        bg_png = data_url + '/' + self.modifier + '/bg' + '.png'
        if Path(bg_png).is_file():
            bg = mpimg.imread(bg_png)
        else:
            bg = np.zeros((10, 10))
            bg += 1e-4
        return bg

    def load_bg(self):
        bg_png = data_url + '/' + self.modifier + '/bg.tif'
        if Path(bg_png).is_file():
            bg = Image.open(bg_png)
        else:
            bg = self.load_bg_png()
        return bg

    def get_xy(self):
        return self.X, self.y, self.col_names

    def get_y(self):
        return self.y


def preprocess_input(train_loader, test_loader, lnglat=True):
    X_train = train_loader.X
    y_train = train_loader.y
    col_names = train_loader.col_names

    X_test = test_loader.X
    lnglat = 0 if lnglat else 2  # 0 includes lng,lat. 2 is excluding.
    trainLngLat = X_train[:, :2]
    testLngLat = X_test[:, :2]

    X_train = X_train[y_train != 0]
    X_train = X_train[:, lnglat:]
    y_train = y_train[y_train != 0]

    test_nans = np.isnan(X_test).any(axis=1)
    X_test = X_test[~test_nans]
    X_test_feats = X_test[:, lnglat:]
    col_names = col_names[lnglat:]
    return X_train, y_train, X_test_feats, trainLngLat, testLngLat, test_nans, col_names


# def preprocess_input(train_loader, test_loader, X_train, y_train, X_test, col_names, lnglat=True):
#     lnglat = 0 if lnglat else 2  # 0 includes lng,lat. 2 is excluding.
#     trainLngLat = X_train[:, :2]
#     testLngLat = X_test[:, :2]
#
#     X_train = X_train[y_train != 0]
#     X_train = X_train[:, lnglat:]
#     y_train = y_train[y_train != 0]
#
#     test_nans = np.isnan(X_test).any(axis=1)
#     X_test = X_test[~test_nans]
#     X_test_feats = X_test[:, lnglat:]
#     col_names = col_names[lnglat:]
#     return X_train, y_train, X_test_feats, trainLngLat, testLngLat, test_nans, col_names
