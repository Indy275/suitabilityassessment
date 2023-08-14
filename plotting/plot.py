from plotting import plot_data, plot_pred


def plot_y(loader, bg, ref_std):
    plot_data.plot_y(loader, bg, ref_std)


def plot_f_importances(coef, names):
    plot_pred.plot_f_importances(coef, names)


def adjust_predictions(y, digitize=False, sigmoidal_tf=False):
    return plot_pred.adjust_predictions(y, digitize=digitize, sigmoidal_tf=sigmoidal_tf)


def plot_prediction(y_pred, size, figname=None, title='', train_labs=None, contour=True, bg=None, savefig=True):
    plot_pred.plot_prediction(y_pred, size, fig_name=figname, title=title, train_labs=train_labs, contour=contour,
                              bg=bg, savefig=savefig)


def plot_loss(loss):
    plot_pred.plot_loss(loss)


def set_colmap():
    return plot_pred.set_colmap()
