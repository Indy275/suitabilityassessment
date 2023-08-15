import configparser

import geopandas
import numpy as np
import pandas as pd
import rasterio
from rasterio.plot import show
from geocube.api.core import make_geocube

from data_util import data_loader
from plotting import plot


from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
config = configparser.ConfigParser()
config.read('config.ini')

data_url = config['DEFAULT']['data_url']
fig_url = config['DEFAULT']['fig_url']
plot_data = int(config['PLOTTING']['plot_data'])
plot_pred = int(config['PLOTTING']['plot_prediction'])
plot_feature_importance = int(config['PLOTTING']['plot_feature_importance'])
plot_features = int(config['PLOTTING']['plot_features'])
digitize = int(config['PLOTTING']['digitize_prediction'])
sigmoidal_tf = int(config['PLOTTING']['sigmoid_prediction'])


def save_imp(imp, names, model, modifier):
    imp = np.abs(imp)
    new_imp = imp / np.max(imp)
    impFrame = pd.DataFrame(data=dict({'feature': names, 'importance': new_imp}))
    impFrame.to_csv(data_url + '/' + modifier + "/" + model + "_fimp.csv", index=False)


def train(X_train, y_train, model):
    if model == 'gbr':
        predictor = GradientBoostingRegressor()
        predictor.fit(X_train, y_train)
        feature_imp = predictor.feature_importances_

    elif model == 'svm':
        kernel = 'linear'  # 'linear'  'rbf'
        predictor = OneClassSVM(kernel=kernel)
        predictor.fit(X=X_train)
        feature_imp = predictor.coef_[0]

    else:
        print("Invalid model; should be one of ['gbr','svm']")
        return

    return predictor, feature_imp


def predict(LngLat, X_test, test_nans, model, predictor):
    y_preds = np.zeros((test_nans.shape[0], 3))  # (:, lon, lat, y_test)
    y_preds[:, 0] = LngLat[:, 0]
    y_preds[:, 1] = LngLat[:, 1]
    y_preds[test_nans, 2] = np.nan
    if X_test.shape[0] == test_nans.shape[0]:
        X_test = X_test[~test_nans]

    if model == 'gbr':
        y_pred = predictor.predict(X_test)

    elif model == 'svm':
        y_pred = predictor.score_samples(X_test)
        hist, bins = np.histogram(y_pred, bins=5)
        y_pred = np.digitize(y_pred, bins)

    else:
        print("Invalid model; should be one of ['gbr','svm']")
        return

    y_preds[~test_nans, -1] = y_pred
    return y_preds


def evaluate(test_mod, predictor, test_indices):
    eval_loader = data_loader.DataLoader(test_mod, ref_std='expert_ref')
    x_eval, y_eval, eval_lnglat, test_col_names = eval_loader.preprocess_input()
    eval_nans = np.isnan(x_eval).any(axis=1)
    x_eval = x_eval[:, test_indices]

    y_pred = predict(eval_lnglat, x_eval, eval_nans, model='gbr', predictor=predictor)
    mse = mean_squared_error(y_eval, y_pred[:, -2])
    print("Test MSE: {:.4f}".format(mse))


def plot_feature(data_loader, model, predictor, ind):
    x_train, y_train, lnglat, col_names = data_loader.preprocess_input()
    # orig_df = data_loader.load_orig_df()[:, ind]
    # orig_df = data_loader.X_orig[~data_loader.nans][:, ind]
    x_train = x_train[:, ind]
    col_names = np.array(col_names)[ind]

    # print(f'{orig_df.shape=} {x_train.shape=}, {y_train.shape=} {y_train=}')

    n_samples = 200

    for feature in range(len(ind)):
        nullvals = np.min(x_train, axis=0)
        x_eval = np.tile(nullvals, (n_samples, 1))

        minval = np.min(x_train[:, feature])
        maxval = np.max(x_train[:, feature])
        x_feature = np.linspace(minval, maxval, n_samples)

        # minval_orig = np.min(orig_df[:, feature])
        # maxval_orig = np.max(orig_df[:, feature])
        # x_orig = np.linspace(minval_orig, maxval_orig, n_samples)

        x_eval[:, feature] = x_feature
        test_nans = np.array([False] * n_samples)
        lnglat = np.zeros((n_samples, 2))
        y_pred = predict(lnglat, x_eval, test_nans.T, model=model, predictor=predictor)
        y_pred = y_pred[:, 2]

        if model == 'svm':  # svm provides score rather than class. Force hyperplane between 0-1
            mms = MinMaxScaler()
            y_pred = mms.fit_transform(y_pred.reshape(-1, 1))

        # plt.scatter(orig_df[:, feature], y_train, marker='x')
        # plt.plot(x_orig, y_pred, color='orange')
        plt.scatter(x_train[:, feature], y_train, marker='x')
        plt.plot(x_feature, y_pred, color='orange')
        plt.xlabel(col_names[feature])
        plt.ylabel("Suitability score")
        plt.show()


def run_model(train_mod, test_mod, model, ref_std):
    x_subset = False
    contour = True

    train_data = data_loader.DataLoader(train_mod, ref_std=ref_std)
    test_data = data_loader.DataLoader(test_mod, ref_std='testdata')

    X_train, y_train, train_lnglat, train_col_names = train_data.preprocess_input()
    X_test, test_nans, test_lnglat, test_size, test_col_names = test_data.preprocess_input()

    # Due to strange labels, this appears necessary
    train_nans = np.isnan(X_train).any(axis=1)
    X_train = X_train[~train_nans, :]

    bg_test = test_data.load_bg()
    assert train_col_names == test_col_names

    if plot_data:
        bg_train = train_data.load_bg()
        plot.plot_y(train_data, bg_train, ref_std=ref_std)

    x_indices = list(range(X_train.shape[1]))
    if x_subset:
        x_indices = [0, 1, 3]
        X_train = X_train[:, x_indices]
        X_test = X_test[:, x_indices]
        train_col_names = np.array(train_col_names)[x_indices]

    print(f'Training model. Train shape: {X_train.shape}. Test shape: {X_test.shape}')
    print(f'Train variables: {train_col_names}')

    predictor, feature_imp = train(X_train, y_train, model)
    y_preds = predict(test_lnglat, X_test, test_nans, model, predictor)

    y_preds[~test_nans, 2] = plot.adjust_predictions(y_preds[~test_nans, 2], digitize=digitize, sigmoidal_tf=sigmoidal_tf)

    if plot_pred:
        fig_name = fig_url + '/' + test_mod + '_' + train_mod + '_' + model
        fig_title = f'{model} predictions trained on {train_mod} labels'
        train_labs = None
        if train_mod == test_mod and ref_std == 'expert_ref':
            train_labs = np.column_stack([train_lnglat[:, 0], train_lnglat[:, 1], y_train])
        plot.plot_prediction(y_preds, test_size, fig_name, title=fig_title, train_labs=train_labs, contour=contour, bg=bg_test)

    if plot_feature_importance:
        plot.plot_f_importances(feature_imp, train_col_names)
        save_imp(feature_imp, train_col_names, model, test_mod)

    if test_mod.startswith(('ws', 'oc')):  # test labels exist only for these data sets
        evaluate(test_mod, predictor,x_indices)

    if plot_features:
        plot_feature(train_data, model, predictor, x_indices)

