import configparser
import time
import numpy as np
import pandas as pd

from data_util import data_loader
from plotting import plot

config = configparser.ConfigParser()
config.read('config.ini')

data_url = config['DEFAULT']['data_url']
fig_url = config['DEFAULT']['fig_url']
cluster = config['EXPERT_SETTINGS']['cluster']
digitize = int(config['PLOTTING']['digitize_prediction'])
sigmoidal_tf = int(config['PLOTTING']['sigmoid_prediction'])
coords_as_features = config.getboolean('DATA_SETTINGS', 'coords_as_features')


def predict(unweighted_df, nans, weights):
    weighted_df = np.zeros((unweighted_df.shape[0], 1))
    for coord in range(weighted_df.shape[0]):
        val = 0
        for feature in range(unweighted_df.shape[1]):
            val += unweighted_df[coord, feature] * weights[feature] * -1
        weighted_df[coord] = val / unweighted_df.shape[1]
    weighted_df += abs(min(weighted_df))  # ensure all score are positive

    weighted_df = (weighted_df - weighted_df.mean()) / weighted_df.std()  # standardisation
    weighted_df *= (5 * -1 / weighted_df.min())
    weighted_df[nans] = np.nan
    return np.squeeze(weighted_df)


def run_model(mod):
    contour = True

    start_time = time.time()
    weights = pd.read_csv(data_url + f'/factorweights_{cluster.upper()}.csv')
    weights = list(weights['IQR'])
    weights = [w/max(weights) for w in weights] * -1
    data = data_loader.DataLoader(mod, ref_std='testdata')
    unweighted_df, nans, lnglat, size, col_names = data.preprocess_input()
    if coords_as_features:
        unweighted_df = unweighted_df[:, 2:]  # Always exclude longitude and latitude: importance not measured

    bg = data.load_bg()
    unweighted_df[nans] = 0
    preds = np.zeros((nans.shape[0], 3))
    preds[:, 0] = lnglat[:, 0]  # test longitude
    preds[:, 1] = lnglat[:, 1]  # test latitude
    preds[:, 2] = predict(unweighted_df, nans, weights)
    preds[~nans, 2] = plot.adjust_predictions(preds[~nans, 2], digitize=digitize, sigmoidal_tf=sigmoidal_tf)

    fig_name = '/' + cluster
    if mod == 'noordhollandHiRes': fig_name += '_NH-HR'
    elif mod == 'noordholland': fig_name += '_NH'
    if digitize: fig_name += '_dig'
    if sigmoidal_tf: fig_name += '_sig'
    print(f'Plot took {time.time() - start_time} seconds to create {fig_name} map')
    fig_title = 'Suitability map of Noord Holland'
    if mod in ['ws', 'oc']:  # plot train labels on top of prediction
        loader = data_loader.DataLoader(mod, ref_std='expert_ref')
        train_labs = np.column_stack([loader.lnglat[:, 0], loader.lnglat[:, 1], loader.y])
        plot.plot_prediction(preds, data.size, fig_url+fig_name, title=fig_title, train_labs=train_labs, contour=contour,
                             bg=bg, savefig=True)
    else:
        plot.plot_prediction(preds, data.size, fig_url+fig_name, title=fig_title, contour=contour, bg=bg, savefig=True)
