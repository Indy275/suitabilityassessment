import configparser

import numpy as np
import pandas as pd

from data_util import data_loader
from plotting import plot

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import OneClassSVM

config = configparser.ConfigParser()
config.read('config.ini')

data_url = config['DEFAULT']['data_url']
plot_data = int(config['PLOTTING']['plot_data'])
plot_pred = int(config['PLOTTING']['plot_pred'])
plot_feature_importance = int(config['PLOTTING']['plot_feature_importance'])


def save_imp(imp, names, model, modifier):
    imp = np.abs(imp)
    new_imp = imp / np.max(imp)
    impFrame = pd.DataFrame(data=dict({'feature': names, 'importance': new_imp}))
    impFrame.to_csv(data_url + '/' + modifier + "/" + model + "_fimp.csv", index=False)


def preprocess_input(X_train, y_train, X_test, col_names):
    lonlat = 2  # 0 if include lonlat; 2 if exclude
    # n_feats = X_train.shape[-1]
    testLngLat = X_test[:, :2]

    X_train = X_train[y_train != 0]
    # X_train = X_train.reshape((-1, n_feats))
    X_train = X_train[:, lonlat:]
    y_train = y_train[y_train != 0]

    test_nans = np.isnan(X_test).any(axis=1)
    X_test = X_test[~test_nans]#.reshape((-1, n_feats))
    X_test_feats = X_test[:, lonlat:]
    col_names = col_names[lonlat:-1]
    return X_train, y_train, X_test_feats, testLngLat, test_nans, col_names


def train(X_train, y_train, model):
    if model == 'gbr':
        predictor = GradientBoostingRegressor()
        predictor.fit(X_train, y_train)
    elif model == 'svm':
        # posfeat = [feat for feat, lab in zip(X_train, y_train) if lab > 0]
        kernel = 'linear'  # 'linear'  'rbf'
        predictor = OneClassSVM(kernel=kernel)
        predictor.fit(X=X_train)
    else:
        print("Invalid model; should be one of ['gbr','svm']")
        return
    return predictor


def predict(LngLat, X_test, test_nans, test_mod, test_size, model, predictor, col_names, bg, fig_name):
    y_preds = np.zeros((test_nans.shape[0], 3))  # (:, lon, lat, y_test)
    y_preds[:, 0] = LngLat[:, 0]
    y_preds[:, 1] = LngLat[:, 1]
    y_preds[test_nans, :] = np.nan

    if model == 'gbr':
        y_pred = predictor.predict(X_test)
        feature_imp = predictor.feature_importances_
    elif model == 'svm':
        y_pred = predictor.score_samples(X_test)
        feature_imp = predictor.coef_[0]
    else:
        print("Invalid model; should be one of ['gbr','svm']")
        return

    y_preds[~test_nans, -1] = y_pred
    pd.DataFrame(y_preds).to_csv(data_url + '/' + test_mod + "/" + model + "_pred.csv")

    if plot_pred:
        plot.plot_prediction(y_preds, test_size, fig_name, bg=bg)

    if plot_feature_importance:
        plot.plot_f_importances(feature_imp, col_names)
        save_imp(feature_imp, col_names, model, test_mod)
    return y_preds


def eval_model(y_test, y_preds, test_vals):
    print(y_test[test_vals], y_preds[test_vals, -1])
    mse = mean_squared_error(y_test[test_vals], y_preds[test_vals, -1])
    print("Test MSE: {:.4f}".format(mse))


def run_model(train_mod, test_mod, model, train_size, test_size, ref_std):
    X_train, y_train, train_col_names = data_loader.load_xy(train_mod, model=ref_std)
    X_test, test_col_names = data_loader.load_x(test_mod)
    assert train_col_names == test_col_names
    bg_test = data_loader.load_bg(test_mod)

    if plot_data:
        bg_train = data_loader.load_bg(train_mod)
        plot.plot_y(y_train, bg_train, bg_test, train_size, test_size)

    fig_url = f'C://Users/indy.dolmans/OneDrive - Nelen & Schuurmans/Pictures/maps/'
    fig_name = fig_url + test_mod + '_' + train_mod + '_' + model

    X_train, y_train, X_test, LngLat, test_nans, col_names = preprocess_input(X_train, y_train, X_test, train_col_names)
    predictor = train(X_train, y_train, model)
    y_preds = predict(LngLat, X_test, test_nans, test_mod, test_size, model, predictor, col_names, bg_test, fig_name)

    if data_loader.load_xy(test_mod, model=ref_std):
        X_test, y_test, _ = data_loader.load_xy(test_mod, model=ref_std)
        test_vals = y_test != 0
        eval_model(y_test, y_preds, test_vals)
