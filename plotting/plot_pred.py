import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from textwrap import wrap
import matplotlib as mpl


def set_colmap(img):
    colmap = LinearSegmentedColormap.from_list('colmap2', [[1, 0, 0, 1], [145 / 255, 210 / 255, 80 / 255, 1]])
    try:
        plt.register_cmap(cmap=colmap)
    except:
        pass
    plt.set_cmap(colmap)
    img.cmap.set_bad('tab:blue')


def plot_f_importances(coef, names):
    imp = coef
    imp = np.abs(imp)
    imp, names = zip(*sorted(zip(imp, names)))
    names = ['\n'.join(wrap(x, 20)) for x in names]
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.subplots_adjust(left=0.4)
    plt.show()


def plot_prediction(y_pred, height, digitize_pred=False, sigmoid_pred=False, ax=None):
    X1 = y_pred[:, 0]
    X2 = y_pred[:, 1]
    y_pred = y_pred[:, 2]
    y_pred = np.ma.masked_invalid(y_pred)


    if digitize_pred:
        y_pred = np.digitize(y_pred, np.quantile(y_pred, [0, 0.2, 0.4, 0.6, 0.8, 1.0]))  # digitized values: 5 classes
    elif sigmoid_pred:
        sigmoid = lambda x: 1 / (1 + np.exp(-.5 * (x - .5)))
        y_pred = sigmoid(y_pred)
        print("sigmoid transformed predictions are in range [{0:.3f},{1:.3f}]".format(np.nanmin(y_pred),
                                                                                      np.nanmax(y_pred)))
    else:
        print("predictions are in range [{0:.3f},{1:.3f}]".format(np.nanmin(y_pred), np.nanmax(y_pred)))

    if ax is None:
        fig, ax = plt.subplots()
    # img = ax.imshow(y_pred.reshape((height, -1)), vmin=np.nanmin(y_pred), vmax=np.nanmax(y_pred))  # Should this be swapped? Even though it doesn't matter for square img
    img = ax.imshow(y_pred.reshape((height, -1)), vmin=np.nanmin(y_pred),
                     vmax=np.nanmax(y_pred), extent=[np.nanmin(X1), np.nanmax(X1), np.nanmin(X2), np.nanmax(X2)], aspect='auto')
    set_colmap(img)
    plt.show()  # Show original prediction


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
