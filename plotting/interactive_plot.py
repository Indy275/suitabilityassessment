import os.path
import time
import configparser

import numpy as np
import pandas as pd

from data_util import load_data

import plotly.express as px
from dash import Dash, dcc, html, Input, Output, ctx

config = configparser.ConfigParser()
parent = os.path.dirname
config.read(os.path.join(parent(parent(__file__)), 'config.ini'))

data_url = config['DEFAULT']['data_url']

sign = [-1, -1, -1, -1, -1]  # 1 if higher is better; -1 if lower is better


class DashApp():
    def __init__(self, modifier, w, h, w_model=None):
        self.app = Dash(__name__)
        if w_model:
            weights_init = pd.read_csv(data_url + "/" + modifier + "/" + w_model + '_fimp.csv')
            weights_init = weights_init['importance']
        else:
            weights_init = np.repeat([1], 5)
        unweighted_df, df_orig, col_names = load_data.load_xy(modifier)
        unweighted_df[np.isnan(unweighted_df)] = 0
        df_orig[np.isnan(df_orig)] = 0

        feature0 = df_orig[:, 0].reshape((h, w))
        feature1 = df_orig[:, 1].reshape((h, w))
        feature2 = df_orig[:, 2].reshape((h, w))
        feature3 = df_orig[:, 3].reshape((h, w))
        feature4 = df_orig[:, 4].reshape((h, w))

        self.app.layout = html.Div([
            dcc.Graph(id='graph-with-slider'),
            dcc.Markdown(col_names[0]),
            dcc.Slider(0, 10, 1,
                       value=weights_init[0],
                       tooltip=dict(always_visible=True, placement='bottom'),
                       id='slider1'
                       ),
            dcc.Markdown(col_names[1]),
            dcc.Slider(0, 10, 1,
                       value=weights_init[1],
                       tooltip=dict(always_visible=True, placement='bottom'),
                       id='slider2'
                       ),
            dcc.Markdown(col_names[2]),
            dcc.Slider(0, 10, 1,
                       value=weights_init[2],
                       tooltip=dict(always_visible=True, placement='bottom'),
                       id='slider3'
                       ),
            dcc.Markdown(col_names[3]),
            dcc.Slider(0, 10, 1,
                       value=weights_init[3],
                       tooltip=dict(always_visible=True, placement='bottom'),
                       id='slider4'
                       ),
            dcc.Markdown(col_names[4]),
            dcc.Slider(0, 10, 1,
                       value=weights_init[4],
                       tooltip=dict(always_visible=True, placement='bottom'),
                       id='slider5'
                       )
        ])

        @self.app.callback(
            Output('graph-with-slider', 'figure'),
            [Input('slider1', 'value'),
             Input('slider2', 'value'),
             Input('slider3', 'value'),
             Input('slider4', 'value'),
             Input('slider5', 'value')])
        def update_output(value1, value2, value3, value4, value5):
            start_update_time = time.time()
            weights = [value1, value2, value3, value4, value5]
            weighted_df = np.zeros((unweighted_df.shape[0], 1))
            for coord in range(weighted_df.shape[0]):
                val = 0
                for feature in range(unweighted_df.shape[1]):
                    val += unweighted_df[coord, feature] * weights[feature] * sign[feature]
                weighted_df[coord] = val / unweighted_df.shape[1]
            weighted_df += abs(min(weighted_df))  # ensure all score are positive

            weighted_df = (weighted_df - weighted_df.mean()) / weighted_df.std()  # standardisation
            min_max_val = 5
            weighted_df *= (min_max_val * -1 / weighted_df.min())

            weighted_df = weighted_df.reshape((h, w))
            colmap = [[0.0, 'rgb(255,0,0)'], [1.0, 'rgb(145,210, 80)']]
            fig = px.imshow(weighted_df, labels=dict(color='suitability_score'), color_continuous_scale=colmap)
            updated = ctx.triggered_id
            if updated:
                updated = int(updated[-1]) - 1
                print(
                    "column {} updated; updating took {:.2f}s".format(col_names[updated],
                                                                      time.time() - start_update_time))
            fig.update(data=[{'customdata': np.stack((weighted_df,
                                                      feature0, feature1, feature2, feature3, feature4), axis=-1),
                              'hovertemplate': 'Bouw-geschiktheidsscore: %{customdata[0]:.3g}<br>' +
                                               col_names[0] + ': %{customdata[1]:.3g} m' + ' ({0:.3g}x)<br>'.format(
                                  value1) +
                                               col_names[1] + ': %{customdata[2]:.3g} m' + ' ({0:.3g}x)<br>'.format(
                                  value2) +
                                               col_names[2] + ': %{customdata[3]:.3g} mm/jaar' + ' ({0:.3g}x)<br>'.format(
                                  value3) +
                                               col_names[3] + ': %{customdata[4]:.3g} m' + ' ({0:.3g}x)<br>'.format(
                                  value4) +
                                               col_names[4] + ': %{customdata[5]:.3g} mm' + ' ({0:.3g}x)<br>'.format(
                                  value5) +
                                               '<extra></extra>'}])
            return fig

    def run(self):
        self.app.run_server(debug=True)


# test_mod = 'noordholland'
# test_w = test_h = 1000
# ml_model = 'svm'
#
# plot = DashApp(modifier=test_mod, w=test_w, h=test_h, w_model=ml_model)
# plot.run()