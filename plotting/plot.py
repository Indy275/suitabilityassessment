from plotting import plot_data, plot_pred, gp_plot


def plot_xy(X_train, y_train, X_test, y_test, train_h, train_w, test_h, test_w):
    plot_data.plot_xy(X_train, y_train, X_test, y_test, train_h, train_w, test_h, test_w)


def plot_y(y_train, y_test, bg_train, bg_test, train_w, train_h, test_w, test_h,):
    plot_data.plot_y(y_train, y_test, bg_train, bg_test, train_w, train_h, test_w, test_h,)


def plot_f_importances(coef, names):
    plot_pred.plot_f_importances(coef, names)


def plot_prediction(y_pred, height):
    plot_pred.plot_prediction(y_pred, height)


def plot_gp(Xtest, test_w, test_h, model=None, ax=None):
    gp_plot.plot_pred(Xtest, test_w, test_h, model, ax)


def plot_loss(loss):
    plot_pred.plot_loss(loss)
