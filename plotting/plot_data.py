import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plotting.plot
from shapely.geometry import Point
from collections import Counter


def plot_xy(X_train, y_train, X_test, y_test, train_h, train_w, test_h, test_w):
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(y_train.reshape((train_h, train_w)))
    ax[0, 0].set_title("Train labels")

    ax[0, 1].imshow(X_train.reshape((train_h, train_w, X_train.shape[-1]))[:, :, 0])
    ax[0, 1].set_title("Train feature")

    ax[1, 0].imshow(y_test.reshape((test_h, test_w)))
    ax[1, 0].set_title("Test labels")

    ax[1, 1].imshow(X_test.reshape((test_h, test_w, X_test.shape[-1]))[:, :, 0])
    ax[1, 1].set_title("Test feature")

    plt.tight_layout()
    plt.show()


def plot_row(y, bg, y_t, ax, i, h, w):
    ax[i].imshow(bg, extent=[0, h, 0, w])
    if y_t != 'Test':
        indices = np.nonzero(y.reshape((h, w)))
        ax[i].scatter(indices[1], np.abs(indices[0] - h), s=1, c='r')
        posperc = len(y[y != 0]) / len(y) * 100

        ax[i].set_title("{} labels: {} positive ({:.2f}%)".format(y_t, len(y[y != 0]), posperc))
    ax[i].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False)
    ax[i].yaxis.set_ticklabels([])


def plot_y(y_train, bg_train, bg_test, train_size, test_size):
    fig, ax = plt.subplots(1, 2)

    bg = [bg_train, bg_test]
    y = [y_train, y_train]  # y_test is usually not available and will not be plotted anyway
    y_t = ['Train', 'Test']
    height = [train_size, test_size]
    width = [train_size, test_size]

    for i, (y_i, b, y_t_i, h, w) in enumerate(zip(y, bg, y_t, height, width)):
        plot_row(y_i, b, y_t_i, ax, i, h, w)
    plt.tight_layout()
    plt.show()


def plot_y_expert(mod, lnglat, y, bg):
    fig, ax = plt.subplots()
    cmap = plotting.plot.set_colmap()
    ax.imshow(bg, extent=[4.8281, 5.0457, 52.5892, 52.714], origin='upper', alpha=0.8)

    # df = pd.DataFrame([{'X1':x1, 'X2': x2} for x1, x2 in lnglat[:, :2]])
    # print( df.groupby(['X1','X2']).value_counts())
    # df['Count'] = df.groupby(['X1','X2']).count()
    # print(df.groupby(['X1','X2']).mean())
    # df['y'] = df.groupby(['X1','X2']).mean()
    #
    # print(df['Count'])

    points = [Point(x1, x2) for x1, x2 in lnglat[:, :2]]
    point_counts = Counter(points)

    df = pd.DataFrame([{'X1': p.x, 'X2': p.y, 'y': y[i],'Count': point_counts[Point(p.x, p.y)]} for i, p in enumerate(points)])
    df_mean = df.groupby(['X1', 'X2'])['y'].mean().to_frame(name='y_mean').reset_index()
    df = df.merge(df_mean, on=['X1', 'X2'], how='left')

    ax.scatter(df['X1'], df['X2'], c=df['y_mean'], cmap=cmap, s=df['Count']*25, edgecolors='black')

    ax.set_title("{} labels: {} data points".format(mod, len(y[y != 0])))
    # ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False)
    plt.show()
