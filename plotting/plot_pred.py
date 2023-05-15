import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from textwrap import wrap
import matplotlib as mpl

def plot_f_importances(coef, names):
    imp = coef
    imp = np.abs(imp)
    imp, names = zip(*sorted(zip(imp, names)))
    names = ['\n'.join(wrap(x, 20)) for x in names]
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.subplots_adjust(left=0.4)
    plt.show()


def plot_prediction(y_pred, height):
    colmap = LinearSegmentedColormap.from_list('colmap2', [[1, 0, 0, 1], [145 / 255, 210 / 255, 80 / 255, 1]])
    try:
        plt.register_cmap(cmap=colmap)
    except:
        pass
    y_pred = np.ma.masked_invalid(y_pred)
    print("predictions are in range [{0:.3f},{1:.3f}]".format(np.nanmin(y_pred), np.nanmax(y_pred)))
    img = plt.imshow(y_pred.reshape((height, -1)), vmin=np.nanmin(y_pred), vmax=np.nanmax(y_pred))  # Should this be swapped? Even though it doesn't matter for square img
    plt.set_cmap(colmap)
    img.cmap.set_bad('tab:blue')
    plt.show()

    sigmoid = lambda x: 1 / (1 + np.exp(-.5*(x-.5)))
    y_pred = sigmoid(y_pred)
    print("logpredictionscores", np.nanmin(y_pred), np.nanmax(y_pred))
    img = plt.imshow(y_pred.reshape((height, -1)), vmin=np.nanmin(y_pred), vmax=np.nanmax(y_pred))
    plt.set_cmap(colmap)
    img.cmap.set_bad('tab:blue')
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
