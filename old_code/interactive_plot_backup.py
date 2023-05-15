import time
import csv
import numpy as np
import pandas as pd
from joblib import load

import configparser

import plotly
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, ctx

config = configparser.ConfigParser()
config.read('config.ini')
data_url = config['DEFAULT']['data_url']

modifier = 'purmer'

sign = [-1, -1, -1, -1, -1]  # 1 if higher is better; -1 if lower is better
weights_init = [1, 1, 1, 1, 1]


def init_score(df, weights, sign, h, w):
    weighted_df = np.zeros((df.shape[0], 1))
    for coord in range(weighted_df.shape[0]):
        val = 0
        for feature in range(df.shape[1]):
            val += df[coord, feature] * weights[feature] * sign[feature]
        weighted_df[coord] = val/df.shape[1]
    return weighted_df.reshape((h,w))


def load_data(modifier):
    # read raster information
    df = pd.read_csv(data_url + '/' + modifier + '.csv')
    col_names = list(df.columns)
    df2 = df.to_numpy()

    X = df2[:, :-1]

    # load and reverse-apply scaler
    ss = load(data_url + '/' + modifier + "scaler.joblib")
    df_orig = ss.inverse_transform(X)

    # load metadata
    with open(data_url + '/' + modifier + '_meta.txt', 'r') as f:
        meta = [t for t in csv.reader(f)]
    w, h = int(meta[0][0]), int(meta[0][1])
    return X, df_orig, w, h, col_names


def create_plot(modifier):
    df, df_orig, w, h, col_names = load_data(modifier)
    weighted_df = init_score(df, weights_init, sign, h, w)
    return weighted_df, df, df_orig, w, h, col_names


weighted_df1, unweighted_df, df_orig, w, h, col_names = create_plot(modifier=modifier)  # For now this does the trick

feature0 = df_orig[:, 0].reshape((h, w))
feature1 = df_orig[:, 1].reshape((h, w))
feature2 = df_orig[:, 2].reshape((h, w))
feature3 = df_orig[:, 3].reshape((h, w))
feature4 = df_orig[:, 4].reshape((h, w))

txt = ''
app = Dash(__name__)

app.layout = html.Div([
    dcc.Markdown(txt, id='updating_markdown'),
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

@app.callback(
    Output('graph-with-slider', 'figure'),
    [Input('slider1', 'value'),
    Input('slider2', 'value'),
    Input('slider3', 'value'),
    Input('slider4', 'value'),
    Input('slider5', 'value')])
def update_output(value1, value2, value3, value4, value5):
    start_update_time = time.time()
    # txt = 'Updating ...'
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
    weighted_df *= (min_max_val*-1 / weighted_df.min())
    # weighted_df = weighted_df.min() * weighted_df

    weighted_df = weighted_df.reshape((h, w))
    colmap = [[0.0, 'rgb(255,0,0)'], [1.0, 'rgb(145,210, 80)']]
    fig = px.imshow(weighted_df, labels=dict(color='suitability_score'), color_continuous_scale=colmap)
    updated = ctx.triggered_id
    if updated:
        updated = int(updated[-1]) - 1
        print("column {} updated; updating took {:.2f}s".format(col_names[updated], time.time() - start_update_time))
        txt = "column {} updated; updating took {:.2f}s".format(col_names[updated], time.time() - start_update_time)
    fig.update(data=[{'customdata': np.stack((weighted_df,
                                              feature0, feature1, feature2, feature3, feature4), axis=-1),
    'hovertemplate': 'Suitability score: %{customdata[0]}<br>' +
                     col_names[0] + ': %{customdata[1]}' + ' ({}x)<br>'.format(value1) +
                     col_names[1] + ': %{customdata[2]}' + ' ({}x)<br>'.format(value2) +
                     col_names[2] + ': %{customdata[3]}' + ' ({}x)<br>'.format(value3) +
                     col_names[3] + ': %{customdata[4]}' + ' ({}x)<br>'.format(value4) +
                     col_names[4] + ': %{customdata[5]}' + ' ({}x)<br>'.format(value5) +
                     '<extra></extra>'}])
    return fig#, txt


if __name__ == '__main__':
    app.run_server(debug=True)