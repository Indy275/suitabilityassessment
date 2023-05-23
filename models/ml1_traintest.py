import configparser

import numpy as np
import pandas as pd

from data_util import load_data
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
    new_imp = imp/np.max(imp)
    impFrame = pd.DataFrame(data=dict({'feature': names, 'importance': new_imp}))
    impFrame.to_csv(data_url + '/' + modifier + "/" + model + "_fimp.csv", index=False)


def train_predict(X_train, y_train, X_test, y_test, test_w, test_h, model, modifier, col_names, plot_fimp=0):
    n_feats = X_train.shape[-1]
    test_nans = np.isnan(X_test)

    train_vals = y_train != 0
    test_vals = y_test != 0

    X_test = X_test[~test_nans]
    X_train = X_train[train_vals]
    y_train = y_train[train_vals]
    X_train = X_train.reshape((-1, n_feats))
    X_test = X_test.reshape((-1, n_feats))

    y_preds = np.zeros(y_test.shape)
    y_preds[test_nans[:, 0]] = np.nan

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
            plot.plot_f_importances(reg.feature_importances_, col_names[:-1])
            save_imp(reg.feature_importances_, col_names[:-1], model, modifier)
    elif model == 'svm':
        posfeat = [feat for feat, lab in zip(X_train, y_train) if lab > 0]
        kernel = 'linear'  # 'linear'
        svm = OneClassSVM(kernel=kernel)
        svm.fit(X=posfeat)
        y_pred = svm.predict(X_test)
        y_pred2 = svm.score_samples(X_test)
        if plot_fimp and kernel == 'linear':
            plot.plot_f_importances(svm.coef_[0], col_names[:-1])
            save_imp(svm.coef_[0], col_names[:-1], model, modifier)
    else:
        print("Invalid model; should be one of ['gbr','svm']")
        y_pred = np.zeros((test_h * test_w))
    y_preds[~test_nans[:, 0]] = y_pred
    mse = mean_squared_error(y_test[test_vals], y_preds[test_vals])
    print("Test MSE: {:.4f}".format(mse))
    pd.DataFrame(y_preds).to_csv(data_url + '/' + modifier + "/" + model + "_pred.csv")
    if plot_pred:
        plot.plot_prediction(y_preds, test_h)
        if model == 'svm':
            y_preds[~test_nans[:, 0]] = y_pred2
            plot.plot_prediction(y_preds, test_h)


def run_model(train_mod, test_mod, model, train_w, train_h, test_w, test_h, ref_std):

    X_train, y_train, X_train_orig, train_col_names = load_data.load_xy(train_mod, ref_std)
    X_test, y_test, X_test_orig, test_col_names = load_data.load_xy(test_mod, ref_std)
    assert train_col_names == test_col_names

    if plot_data:
        # print("Train features:", X_train.shape, pd.DataFrame(X_train).describe())
        # print("Train labels:", y_train.shape, pd.DataFrame(y_train).describe())
        # print("Of the training samples, {} are positive ({}%)".format(sum(y_train), sum(y_train) / len(y_train) * 100))

        # print("Test features:", X_test.shape, pd.DataFrame(X_test).describe())
        # print("Test labels:", y_test.shape, pd.DataFrame(y_test).describe())
        # print("Of the test samples, {} are positive ({}%)".format(sum(y_test), sum(y_test) / len(y_test) * 100))

        bg_train = load_data.load_bg(train_mod)
        bg_test = load_data.load_bg(test_mod)
        plot.plot_y(y_train, y_test, bg_train, bg_test, train_w, train_h, test_w, test_h)

    train_predict(X_train, y_train, X_test, y_test, test_w, test_h, model, test_mod, test_col_names,
                  plot_fimp=plot_feature_importance)
