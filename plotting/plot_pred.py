import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from textwrap import wrap
import matplotlib as mpl
import pandas as pd

from shapely.geometry import Point
from collections import Counter


def set_colmap():
    colmap = LinearSegmentedColormap.from_list('colmap2', [[1, 0, 0, 1],
                                                           [211 / 255, 146 / 255, 58 / 255],
                                                           [145 / 255, 210 / 255, 80 / 255, 1]])  # label
    # colmap = LinearSegmentedColormap.from_list('colmap2', [[145 / 255, 210 / 255, 80 / 255, 1], [1, 0, 0, 1]])  # msk
    try:
        plt.register_cmap(cmap=colmap)
    except:
        pass
    plt.set_cmap(colmap)
    # colmap = mpl.cm.get_cmap('viridis')
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


def adjust_predictions(y, digitize, sigmoidal_tf):
    if digitize:
        n_quantiles = 11
        plt.hist(y, bins=30, edgecolor='k')
        plt.title("Histogram before applying digitization")
        plt.show()
        quantiles = np.linspace(0, 1, n_quantiles)
        y = np.digitize(y, np.nanquantile(y, quantiles))
        plt.hist(y, edgecolor='k')
        plt.title("Histogram after applying digitization")
        plt.xticks(range(n_quantiles), [f' {i + 1}' for i in range(n_quantiles)])
        plt.show()
    elif sigmoidal_tf:
        plt.hist(y, bins=30, edgecolor='k')
        plt.title("Histogram before applying \n sigmoid transformation")
        plt.show()
        sigmoid = lambda x: 1 / (1 + np.exp(-.5 * (x - .5)))
        y = sigmoid(y)
        plt.hist(y, edgecolor='k')
        plt.title("Histogram after applying \n sigmoid transformation")
        plt.show()
    return y


def plot_prediction(y_preds, test_size, fig_name=None, title='', train_labs=None, contour=True, bg=None, savefig=True):
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
    plt.imshow(bg, origin='upper', aspect=ratio,
               extent=[np.nanmin(X1), np.nanmax(X1), np.nanmin(X2), np.nanmax(X2)])
    if not contour:
        plt.imshow(y_preds[:, 2].reshape((test_size, test_size)), alpha=0.65, cmap=cmap,
                   extent=[np.nanmin(X1), np.nanmax(X1), np.nanmin(X2), np.nanmax(X2)], aspect=ratio)
    else:
        plt.contourf(X1[:test_size], X2[::test_size], y_preds[:, 2].reshape((test_size, test_size)),
                     cmap=cmap, origin='upper', alpha=0.65)
        if savefig:
            fig_name += '_cont'

    cbar = plt.colorbar()
    mn = np.nanmin(y_preds[:, 2])  # colorbar min value
    mx = np.nanmax(y_preds[:, 2])  # colorbar max value
    cbar.set_ticks([mn, mx])
    cbar.set_ticklabels(['Less suitable', 'More suitable'])

    if train_labs is not None:
        points = [Point(x1, x2) for x1, x2 in train_labs[:, :2]]
        point_counts = Counter(points)
        y = train_labs[:, 2]
        df = pd.DataFrame(
            [{'X1': p.x, 'X2': p.y, 'y': y[i], 'Count': point_counts[Point(p.x, p.y)]} for i, p in enumerate(points)])
        df_mean = df.groupby(['X1', 'X2'])['y'].mean().to_frame(name='y_mean').reset_index()
        df = df.merge(df_mean, on=['X1', 'X2'], how='left')
        df.sort_values(['y_mean'], inplace=True)

        plt.scatter(df['X1'], df['X2'], c=df['y'], cmap=cmap, s=df['Count'] * 25, edgecolors='black')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(title)
    if savefig:
        plt.savefig(fig_name, bbox_inches='tight')
    plt.show()


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
