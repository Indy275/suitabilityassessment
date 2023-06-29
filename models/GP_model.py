import configparser
import numpy as np

from data_util import data_loader
from plotting import plot

import torch
from gpytorch.models import ExactGP
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal

config = configparser.ConfigParser()
config.read('config.ini')

plot_data = int(config['PLOTTING']['plot_data'])
plot_pred = int(config['PLOTTING']['plot_pred'])
plot_feature_importance = int(config['PLOTTING']['plot_feature_importance'])


def set_hypers():
    nu = float(config['MODEL_PARAMS_GP']['nu'])
    num_steps = int(config['MODEL_PARAMS_GP']['num_steps'])
    lr = float(config['MODEL_PARAMS_GP']['lr'])
    return nu, num_steps, lr


class ExactGPModel(ExactGP):
    def __init__(self, X_train, y_train, kernel, likelihood):
        super(ExactGPModel, self).__init__(X_train, y_train, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


def preprocess_input(X_train, y_train, X_test, col_names):
    lonlat = 2  # 0 if include lonlat; 2 if exclude
    # n_feats = X_train.shape[-1]
    testLngLat = X_test[:, :2]

    X_train = X_train[y_train != 0]
    # X_train = X_train.reshape((-1, n_feats))
    X_train = X_train[:, lonlat:]
    y_train = y_train[y_train != 0]

    test_nans = np.isnan(X_test).any(axis=1)
    X_test = X_test[~test_nans]#.reshape((-1, n_feats))
    X_test_feats = X_test[:, lonlat:]
    col_names = col_names[lonlat:-1]
    return X_train, y_train, X_test_feats, testLngLat, test_nans, col_names


def train(train_x, train_y, kernel, num_steps=500, lr=0.01):
    likelihood = GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, kernel, likelihood)

    # Set trainable
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = ExactMarginalLogLikelihood(likelihood, model)

    losses = []
    for i in range(num_steps):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        losses.append(loss)
        if i % 50 == 0:
            print('Iter %d/%d - Loss: %.3f   noise: %.3f' % (
                i + 1, num_steps, loss.item(),
                model.likelihood.noise.item()
            ))
        optimizer.step()

    print('Training completed - Loss: %.3f   noise: %.3f ' % (
        losses[-1].item(),
        model.likelihood.noise.item()
    ))
    with torch.no_grad():
        ls = model.covar_module.base_kernel.lengthscale.detach().numpy()

    return model, likelihood, losses, ls[0]


def predict(LngLat, X_test, test_nans, model, likelihood):
    y_preds = np.zeros((test_nans.shape[0], 4))  # (:, lon, lat, y_pred, y_var)
    y_preds[:, 0] = LngLat[:, 0]  # test y
    y_preds[:, 1] = LngLat[:, 1]  # test x
    y_preds[test_nans, 2] = np.nan
    y_preds[test_nans, 3] = np.nan

    # test phase
    model.eval()
    likelihood.eval()
    f_preds = model(X_test)
    y_pred = likelihood(model(X_test))

    f_mean = f_preds.mean
    f_var = f_preds.variance
    # f_covar = f_preds.covariance_matrix
    y_mean = y_pred.mean

    with torch.no_grad():
        y_preds[~test_nans, 2] = y_mean.numpy()
        y_preds[~test_nans, 3] = f_var.numpy()
    return y_preds


def run_model(train_mod, test_mod, test_size):
    X_train, y_train, train_col_names = data_loader.load_xy(train_mod, model='expert_ref')
    X_test, test_col_names = data_loader.load_x(test_mod)
    assert train_col_names == test_col_names
    bg_test = data_loader.load_bg(test_mod)

    X_train, y_train, X_test, LngLat, test_nans, col_names = preprocess_input(X_train, y_train, X_test, train_col_names)
    n_feats = X_train.shape[1]

    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).float()
    X_test = torch.tensor(X_test).float()

    nu, num_steps, lr = set_hypers()
    kernels = {'Matern': ScaleKernel(MaternKernel(nu=nu, ard_num_dims=n_feats)),
               'RBF': ScaleKernel(RBFKernel(ard_num_dims=n_feats))}
    for k_name, kernel in kernels.items():
        model, likelihood, losses, ls = train(X_train, y_train, kernel, num_steps=num_steps, lr=lr)
        y_preds = predict(LngLat, X_test, test_nans, model, likelihood)

        if plot_feature_importance:
            print('ARD lengthscale: ', ls)
            f_imp = [1 / x for x in ls]  # longer length scale means less important
            f_imp = np.array(f_imp) / np.sum(f_imp)  # normalize length scales
            plot.plot_f_importances(f_imp, col_names)

        if plot_pred:
            fig_url = 'C://Users/indy.dolmans/OneDrive - Nelen & Schuurmans/Pictures/maps/'
            fig_name = fig_url + test_mod + '_' + train_mod + '_GP' + k_name
            plot.plot_contour(y_preds, test_size, fig_name, bg_test)

            losses = [loss.detach().numpy() for loss in losses]
            plot.plot_loss(losses)
