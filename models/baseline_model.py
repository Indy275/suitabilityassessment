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


def train(X_train, y_train, model):
    if model == 'gbr':
        predictor = GradientBoostingRegressor()
        predictor.fit(X_train, y_train)
        feature_imp = predictor.feature_importances_
    elif model == 'svm':
        # posfeat = [feat for feat, lab in zip(X_train, y_train) if lab > 0]
        kernel = 'linear'  # 'linear'  'rbf'
        predictor = OneClassSVM(kernel=kernel)
        predictor.fit(X=X_train)
        feature_imp = predictor.coef_[0]
    else:
        print("Invalid model; should be one of ['gbr','svm']")
        return
    return predictor, feature_imp


def predict(LngLat, X_test, test_nans, test_mod, model, predictor):
    y_preds = np.zeros((test_nans.shape[0], 3))  # (:, lon, lat, y_test)
    y_preds[:, 0] = LngLat[:, 0]
    y_preds[:, 1] = LngLat[:, 1]
    y_preds[test_nans, 2] = np.nan

    if model == 'gbr':
        y_pred = predictor.predict(X_test)

    elif model == 'svm':
        y_pred = predictor.score_samples(X_test)

    else:
        print("Invalid model; should be one of ['gbr','svm']")
        return

    y_preds[~test_nans, -1] = y_pred
    pd.DataFrame(y_preds).to_csv(data_url + '/' + test_mod + "/" + model + "_pred.csv")
    return y_preds


def eval_model(y_test, y_preds, test_vals):
    print(y_test[test_vals], y_preds[test_vals, -1])
    mse = mean_squared_error(y_test[test_vals], y_preds[test_vals, -1])
    print("Test MSE: {:.4f}".format(mse))


def run_model(train_mod, test_mod, model, train_size, test_size, ref_std):
    train_loader = data_loader.DataLoader(train_mod, model)
    test_loader = data_loader.DataLoader(test_mod, model='testdata')

    X_train, y_train, train_lnglat, train_col_names = train_loader.preprocess_input()
    X_test, test_nans, test_lnglat, test_col_names = test_loader.preprocess_input()

    bg_test = test_loader.load_bg()
    assert train_col_names == test_col_names

    if plot_data:
        bg_train = train_loader.load_bg()
        plot.plot_y(y_train, bg_train, bg_test, train_size, test_size)


    # ind = [5]
    # ind = [0, 1]  # Longitude, Latitude
    # X_train = X_train[:, ind]  # temp
    # X_test = X_test[:, ind]  # temp
    # col_names = col_names[ind]  # temp
    print(f'Training model. Train shape: {X_train.shape}. Test shape: {X_test.shape}')
    print(f'Train variables: {train_col_names}')

    predictor, feature_imp = train(X_train, y_train, model)
    y_preds = predict(test_lnglat, X_test, test_nans, test_mod, model, predictor)

    if plot_pred:
        fig_url = f'C://Users/indy.dolmans/OneDrive - Nelen & Schuurmans/Pictures/maps/'
        fig_name = fig_url + test_mod + '_' + train_mod + '_' + model

        if train_mod == test_mod and ref_std == 'expert_ref':
            train_labs = np.column_stack([train_lnglat[:, 0], train_lnglat[:, 1], y_train])
            plot.plot_prediction(y_preds, test_size, fig_name, train_labs, contour=False, bg=bg_test)
        else:
            plot.plot_prediction(y_preds, test_size, fig_name, contour=False, bg=bg_test)

    if plot_feature_importance:
        plot.plot_f_importances(feature_imp, train_col_names)
        save_imp(feature_imp, train_col_names, model, test_mod)

