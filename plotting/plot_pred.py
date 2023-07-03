import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from textwrap import wrap
import matplotlib as mpl


def set_colmap():
    colmap = LinearSegmentedColormap.from_list('colmap2', [[1, 0, 0, 1], [145 / 255, 210 / 255, 80 / 255, 1]])
    try:
        plt.register_cmap(cmap=colmap)
    except:
        pass
    plt.set_cmap(colmap)
    return colmap


def plot_f_importances(coef, names):
    imp = coef
    imp = np.abs(imp)
    imp, names = zip(*sorted(zip(imp, names)))
    names = ['\n'.join(wrap(x, 20)) for x in names]
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.subplots_adjust(left=0.4)
    plt.show()


def plot_prediction(y_preds, test_size, figname, digitize_pred=False, sigmoid_pred=False, ax=None, bg=None):
    X1 = y_preds[:, 1]
    X2 = y_preds[:, 0]
    y_preds[:, 2] = np.ma.masked_invalid(y_preds[:, 2])

    if digitize_pred:  # Broken for NaN data and deprecated
        print("BROKEN AND DEPRECATED: prediction digitization is broken for NaN data; won't be fixed.")
        y_preds[:, 2] = np.digitize(y_preds[:, 2],
                                    np.quantile(y_preds[:, 2], [0, 0.2, 0.4, 0.6, 0.8, 1.0]))  # digitized 5 classes
        print("predictions are digitized and in range [{0:.3f},{1:.3f}]".format(np.nanmin(y_preds[:, 2]),
                                                                                np.nanmax(y_preds[:, 2])))
    elif sigmoid_pred:
        print("BROKEN AND DEPRECATED: sigmoid transformed predictions broken for NaN data; won't be fixed.")
        sigmoid = lambda x: 1 / (1 + np.exp(-.5 * (x - .5)))
        y_preds[:, 2] = sigmoid(y_preds[:, 2])
        print("sigmoid transformed predictions are in range [{0:.3f},{1:.3f}]".format(np.nanmin(y_preds[:, 2]),
                                                                                      np.nanmax(y_preds[:, 2])))
    else:
        print("predictions are in range [{0:.3f},{1:.3f}]".format(np.nanmin(y_preds[:, 2]), np.nanmax(y_preds[:, 2])))

    if ax is None:
        fig, ax = plt.subplots()
    cmap = set_colmap()

    ax.imshow(bg, extent=[np.min(X1), np.max(X1), np.min(X2), np.max(X2)], origin='upper')
    ax.imshow(y_preds[:, 2].reshape((test_size, test_size)), alpha=0.7, cmap=cmap,
              extent=[np.nanmin(X1), np.nanmax(X1), np.nanmin(X2), np.nanmax(X2)])
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    plt.savefig(figname)
    plt.show()


def plot_contour(y_preds, test_size, fig_name, bg=None):
    X1 = y_preds[:, 1]
    X2 = y_preds[:, 0]

    plt.subplots()
    cmap = set_colmap()

    plt.imshow(bg, extent=[np.min(X1), np.max(X1), np.min(X2), np.max(X2)], origin='upper')
    plt.contourf(X1[:test_size], X2[::test_size], y_preds[:, 2].reshape((test_size, test_size)),
                 cmap=cmap, origin='upper', alpha=0.7)  # Plot the mean function as a filled contour
    # plt.contour(X1[:test_size], X2[::test_size], y_preds[:, 3].reshape((test_size, test_size)),
    #             levels=[1.96, 2.58], colors='black', linewidths=0.5, origin='upper', alpha=0.7)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

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
