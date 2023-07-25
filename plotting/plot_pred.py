import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from textwrap import wrap
import matplotlib as mpl
import pandas as pd

from shapely.geometry import Point
from collections import Counter


def set_colmap():
    colmap = LinearSegmentedColormap.from_list('colmap2', [[1, 0, 0, 1], [145 / 255, 210 / 255, 80 / 255, 1]])  # label
    # colmap = LinearSegmentedColormap.from_list('colmap2', [[145 / 255, 210 / 255, 80 / 255, 1], [1, 0, 0, 1]])  # msk
    try:
        plt.register_cmap(cmap=colmap)
    except:
        pass
    plt.set_cmap(colmap)
    return colmap


def translate_names(words):
    to_replace = ['Lng', 'Lat', 'Overstromingsbeeld primaire keringen', 'Overstromingsbeeld regionale keringen',
                  'Bodemdaling Huidig', 'inundatiediepte T100',
                  'Bodemberging bij grondwaterstand gelijk aan streefpeil']
    replace_with = ['Longitude', 'Latitude', 'Flooding risk of primary embankments',
                    'Flooding risk of regional embankments', 'Ground subsidence',
                    'Bottlenecks excessive rainwater', 'Soil water storage capacity']
    for rep, width in zip(to_replace, replace_with):
        words = [w.replace(rep, width) for w in words]
    return words


def plot_f_importances(coef, names):
    imp = coef
    imp = np.abs(imp)
    names = translate_names(names)
    imp, names = zip(*sorted(zip(imp, names)))
    names = ['\n'.join(wrap(x, 20)) for x in names]
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.subplots_adjust(left=0.4)
    plt.show()


def plot_prediction(y_preds, test_size, fig_name, train_labs=None, contour=True, bg=None, savefig=True):
    X1 = y_preds[:, 0]
    X2 = y_preds[:, 1]
    y_preds[:, 2] = np.ma.masked_invalid(y_preds[:, 2])

    print("predictions are in range [{0:.3f},{1:.3f}]".format(np.nanmin(y_preds[:, 2]), np.nanmax(y_preds[:, 2])))

    plt.subplots()
    cmap = set_colmap()
    try:
        width, height = bg.size
    except:
        width, height = bg.shape[0], bg.shape[1]
    ratio = height / width
    plt.imshow(bg, extent=[np.min(X1), np.max(X1), np.min(X2), np.max(X2)], origin='upper', aspect=ratio)
    if not contour:
        plt.imshow(y_preds[:, 2].reshape((test_size, test_size)), alpha=0.55, cmap=cmap,
                   extent=[np.nanmin(X1), np.nanmax(X1), np.nanmin(X2), np.nanmax(X2)], aspect=ratio)
    else:
        plt.contourf(X1[:test_size], X2[::test_size], y_preds[:, 2].reshape((test_size, test_size)),
                     cmap=cmap, origin='upper', alpha=0.55)
        # plt.contour(~np.ma.masked_invalid(y_preds[:, 2].reshape((test_size, test_size))))
        fig_name += '_contour'

    if train_labs is not None:
        points = [Point(x1, x2) for x1, x2 in train_labs[:, :2]]
        point_counts = Counter(points)
        df = pd.DataFrame([
            {'X1': p.x, 'X2': p.y, 'y': train_labs[i, 2], 'Count': point_counts[Point(p.x, p.y)]}
            for i, p in enumerate(points)])

        plt.scatter(df['X1'], df['X2'], c=df['y'], cmap=cmap, s=df['Count'] * 25, edgecolors='black')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    if savefig:
        plt.savefig(fig_name, bbox_inches='tight')
    plt.show()


# def plot_contour(y_preds, test_size, fig_name, bg=None):
#     X1 = y_preds[:, 1]
#     X2 = y_preds[:, 0]
#
#     plt.subplots()
#     cmap = set_colmap()
#
#     plt.imshow(bg, extent=[np.min(X1), np.max(X1), np.min(X2), np.max(X2)], origin='upper')
#     plt.contourf(X1[:test_size], X2[::test_size], y_preds[:, 2].reshape((test_size, test_size)),
#                  cmap=cmap, origin='upper', alpha=0.65)  # Plot the mean function as a filled contour
#     # plt.contour(X1[:test_size], X2[::test_size], y_preds[:, 3].reshape((test_size, test_size)),
#     #             levels=[1.96, 2.58], colors='black', linewidths=0.5, origin='upper', alpha=0.7)
#     plt.xlabel('Longitude')
#     plt.ylabel('Latitude')
#
#     plt.savefig(fig_name, bbox_inches='tight')
#     plt.show()


def plot_colorbar():
    colmap = LinearSegmentedColormap.from_list('colmap2', [[1, 0, 0, 1], [145 / 255, 210 / 255, 80 / 255, 1]])
    try:
        plt.register_cmap(cmap=colmap)
    except:
        pass
    plt.set_cmap(colmap)
    norm = mpl.colors.Normalize(vmin=-5, vmax=5)

    fig = plt.figure(figsize=(8, 3))
    ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
    mpl.colorbar.ColorbarBase(ax1, cmap=colmap,
                              norm=norm,
                              orientation='horizontal')
    plt.show()


def plot_loss(loss):
    plt.plot(loss)
    plt.xlabel("Iterations")
    _ = plt.ylabel("Loss")
    plt.show()
