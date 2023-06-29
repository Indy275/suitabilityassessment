from plotting import plot_data, plot_pred


def plot_xy(X_train, y_train, X_test, y_test, train_h, train_w, test_h, test_w):
    plot_data.plot_xy(X_train, y_train, X_test, y_test, train_h, train_w, test_h, test_w)


def plot_y(y_train, bg_train, bg_test, train_size, test_size):
    plot_data.plot_y(y_train, bg_train, bg_test, train_size, test_size)


def plot_f_importances(coef, names):
    plot_pred.plot_f_importances(coef, names)


def plot_prediction(y_pred, height, figname,digitize_pred=False, sigmoid_pred=False, ax=None, bg=None):
    return plot_pred.plot_prediction(y_pred, height, figname, digitize_pred=digitize_pred, sigmoid_pred=sigmoid_pred, ax=ax, bg=bg)


def plot_contour(y_preds, test_size, fig_name, bg=None):
    plot_pred.plot_contour(y_preds, test_size, fig_name, bg)


def plot_loss(loss):
    plot_pred.plot_loss(loss)


def set_colmap():
    return plot_pred.set_colmap()