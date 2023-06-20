import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde

from data_util import load_data
from plotting import plot

import configparser

config = configparser.ConfigParser()
config.read('config.ini')


def run_model(train_mod, test_h):
    X, Y, _, _ = load_data.load_xy(train_mod, model='hist_buildings')
    X = X[:, :2]
    houses = Y != 0
    X_train = X[houses]
    n_feats = X_train.shape[-1]

    X_test = np.copy(X)
    test_nans = np.isnan(X_test).any(axis=1)
    X_test2 = np.copy(X_test)
    X_test = X_test[~test_nans]
    X_test = X_test.reshape((-1, n_feats))

    y_preds = np.zeros((Y.shape[0],3))  # (:, lon, lat, y_test)
    y_preds[:, 1] = X_test2[:, 0]
    y_preds[:, 0] = X_test2[:, 1]

    y_preds[test_nans, :] = np.nan

    # x, y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    # positions = np.vstack([x.ravel(), y.ravel()])
    kernel = gaussian_kde(X_train.T)
    y_pred = kernel(X_test.T)
    y_preds[~test_nans, -1] = y_pred

    # Contour plot:
    # fig, ax = plt.subplots()
    # ax.scatter(X_train[:, 1], X_train[:, 0], s=1, c='r')
    # ax.contour(y_preds.reshape((test_h, -1)), extent=[np.nanmin(X_test2[:, 1]), np.nanmax(X_test2[:, 1]), np.nanmax(X_test2[:, 0]), np.nanmin(X_test2[:, 0])])
    # plt.show()

    # Smooth plot:
    plot.plot_prediction(y_preds, test_h)
