from plotting import plot_data, plot_pred


def plot_y(loader, bg, ref_std):
    plot_data.plot_y(loader, bg, ref_std)


def plot_f_importances(coef, names):
    plot_pred.plot_f_importances(coef, names)


def plot_prediction(y_pred, height, figname, train_labs=None, contour=True, bg=None, savefig=True):
    plot_pred.plot_prediction(y_pred, height, figname, train_labs=train_labs, contour=contour, bg=bg, savefig=savefig)


def plot_loss(loss):
    plot_pred.plot_loss(loss)


def set_colmap():
    return plot_pred.set_colmap()
