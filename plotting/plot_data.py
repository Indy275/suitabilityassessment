import matplotlib.pyplot as plt
import numpy as np


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


def plot_y(y_train, y_test, bg_train, bg_test, train_w, train_h, test_w, test_h,):
    fig, ax = plt.subplots(2, 2)
    # cmap = colors.Colormap('Set1_r')
    # cmap.set_bad('black')
    ax[0, 0].imshow(y_train.reshape((train_h, train_w)), cmap='Set1_r')
    posperc = len(y_train[y_train != 0]) / len(y_train) * 100
    ax[0, 0].set_title("Train labels: {} positive ({:.3f}%)".format(int(np.nansum(y_train)), posperc))

    ax[0, 1].imshow(bg_train, cmap='Set3')
    ax[0, 1].set_title("Train area")

    ax[1, 0].imshow(y_test.reshape((test_h, test_w)), cmap='Set1_r')
    posperc = len(y_test[y_test != 0]) / len(y_test) * 100
    ax[1, 0].set_title("Test labels: {} positive ({:.3f}%)".format(int(np.nansum(y_test)), posperc))

    ax[1, 1].imshow(bg_test, cmap='Set3')
    ax[1, 1].set_title("Test area")

    plt.tight_layout()
    plt.show()
