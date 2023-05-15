import configparser

import numpy as np
import pandas as pd

from plotting import plot
from data_util import load_data

import gpboost as gpb

config = configparser.ConfigParser()
config.read('config.ini')

data_url = config['DEFAULT']['data_url']
plot_data = int(config['PLOTTING']['plot_data'])
plot_pred = int(config['PLOTTING']['plot_pred'])
plot_feature_importance = int(config['PLOTTING']['plot_feature_importance'])


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


def run_model(modifier, w, h, ref_std):
    features, labels, _, col_names = load_data.load_xy(modifier, ref_std)
    reduced_size = 20
    coords_df = cartesian_product(np.arange(reduced_size), np.arange(reduced_size))
    features = features.reshape((h, w, 5))[:reduced_size, :reduced_size, :].reshape((-1, 5))
    labels = labels.reshape((h, w))[:reduced_size, :reduced_size]#.flatten()
    h = w = reduced_size

    labels[:8, :4] = 1  # No labels are 1 so randomly select some

    if plot_data:
        plot.plot_xy(features, labels, features, labels, h, w, h, w)
    labels = labels.flatten()

    gp_model = gpb.GPModel(gp_coords=coords_df, cov_function="matern", cov_fct_shape=1.5)
    data_train = gpb.Dataset(features, labels)
    params = {'objective': 'regression_l2', 'learning_rate': 0.0001,
              'max_depth': 3, 'min_data_in_leaf': 10,
              'num_leaves': 2 ** 10, 'verbose': 1}
    print(coords_df.shape, features.shape, labels.shape)

    # Training
    bst = gpb.train(params=params, train_set=data_train,
                    gp_model=gp_model, num_boost_round=100)
    gp_model.summary()  # Estimated covariance parameters
    # Make predictions: latent variables and response variable
    pred = bst.predict(data=features, gp_coords_pred=coords_df,
                       predict_var=True, pred_latent=True)
    print(pred)
    if plot_pred:
        plot.plot_prediction(pred['fixed_effect'], h)

    # pred['fixed_effect']: predictions from the tree-ensemble.
    # pred['random_effect_mean']: predicted means of the gp_model.
    # pred['random_effect_cov']: predicted (co-)variances
    pred_resp = bst.predict(data=features, gp_coords_pred=coords_df,
                            predict_var=False, pred_latent=False)
    y_pred = pred_resp['response_mean']  # predicted response mean
    print(pred_resp, y_pred)
    if plot_pred:
        plot.plot_prediction(y_pred, h)
    pd.DataFrame(y_pred).to_csv(data_url + "/predictions_" + modifier + '.csv')
    y_pred[y_pred > 8.114085e-04] = 1  # Best 25% of points for building houses
