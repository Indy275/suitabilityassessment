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


def train_predict(X_train, y_train, X_test, y_test, test_h, model, train_mod, test_mod, col_names, plot_fimp=0):
    lonlat = 2  # 0 if include lonlat; 2 if exclude
    n_feats = X_train.shape[-1]
    X_train = X_train[y_train != 0]
    X_train = X_train.reshape((-1, n_feats))
    X_train = X_train[:, lonlat:]
    y_train = y_train[y_train != 0]

    test_vals = y_test != 0
    test_nans = np.isnan(X_test).any(axis=1)
    X_test2 = np.copy(X_test)
    X_test = X_test[~test_nans]
    X_test = X_test.reshape((-1, n_feats))
    X_test = X_test[:, lonlat:]

    col_names = col_names[lonlat:-1]
    y_preds = np.zeros((y_test.shape[0], 3))  # (:, lon, lat, y_test)
    y_preds[:, 0] = X_test2[:, 0]
    y_preds[:, 1] = X_test2[:, 1]

    y_preds[test_nans, :] = np.nan

    if model == 'gbr':
        reg = GradientBoostingRegressor()
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        # half_preds = pd.DataFrame(y_pred).describe().loc["50%"].tolist()[0]
        # y_pred[y_pred>half_preds] = 1
        # conf_int = pd.DataFrame(y_pred).describe(percentiles=[.9]).loc["90%"].tolist()[0]
        # y_pred /= conf_int
        # y_pred[y_pred > conf_int] = conf_int
        if plot_fimp:
            print(reg.feature_importances_, col_names, model, test_mod)
            plot.plot_f_importances(reg.feature_importances_, col_names)
            save_imp(reg.feature_importances_, col_names, model, test_mod)
    elif model == 'svm':
        posfeat = [feat for feat, lab in zip(X_train, y_train) if lab > 0]
        kernel = 'linear'  # 'linear'  'rbf'
        svm = OneClassSVM(kernel=kernel)
        svm.fit(X=posfeat)
        y_pred = svm.predict(X_test)
        y_pred2 = svm.score_samples(X_test)
        if plot_fimp and kernel == 'linear':
            plot.plot_f_importances(svm.coef_[0], col_names)
            save_imp(svm.coef_[0], col_names, model, test_mod)
    else:
        print("Invalid model; should be one of ['gbr','svm']")
        y_pred = np.repeat(0, y_preds[~test_nans[:, 0]].shape)

    y_preds[~test_nans, -1] = y_pred
    pd.DataFrame(y_preds).to_csv(data_url + '/' + test_mod + "/" + model + "_pred.csv")
    if plot_pred:
        fig_url = 'C://Users/indy.dolmans/OneDrive - Nelen & Schuurmans/Pictures/maps/'
        fig_name = fig_url + test_mod + '_' + train_mod + '_' + model
        plot.plot_prediction(y_preds, test_h, fig_name)

        if model == 'svm':
            y_preds[~test_nans, -1] = y_pred2
            plot.plot_prediction(y_preds, test_h, fig_name + 'scores')
    if sum(test_vals) > 0:  # Only perform MSE test if we have test labels
        print(y_test[test_vals], y_preds[test_vals, -1])
        mse = mean_squared_error(y_test[test_vals], y_preds[test_vals, -1])
        print("Test MSE: {:.4f}".format(mse))


def run_model(train_mod, test_mod, model, train_size, test_size, ref_std):
    X_train, y_train, X_train_orig, train_col_names = data_loader.load_data(train_mod, ref_std)
    X_test, y_test, X_test_orig, test_col_names = data_loader.load_data(test_mod, ref_std)
    assert train_col_names == test_col_names

    if plot_data:
        # print("Train features:", X_train.shape, pd.DataFrame(X_train).describe())
        # print("Train labels:", y_train.shape, pd.DataFrame(y_train).describe())
        # print("Of the training samples, {} are positive ({}%)".format(sum(y_train), sum(y_train) / len(y_train) * 100))

        # print("Test features:", X_test.shape, pd.DataFrame(X_test).describe())
        # print("Test labels:", y_test.shape, pd.DataFrame(y_test).describe())
        # print("Of the test samples, {} are positive ({}%)".format(sum(y_test), sum(y_test) / len(y_test) * 100))

        bg_train = data_loader.load_bg(train_mod)
        bg_test = data_loader.load_bg(test_mod)
        plot.plot_y(y_train, y_test, bg_train, bg_test, train_size, test_size)

    train_predict(X_train, y_train, X_test, y_test, test_size, model, train_mod, test_mod, test_col_names,
                  plot_fimp=plot_feature_importance)
