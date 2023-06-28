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
import matplotlib.pyplot as plt

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


def train(train_x, train_y, model, likelihood, num_steps=500, lr=0.01):
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

    return losses, ls[0]


def run_model(train_mod, test_mod, test_h):
    X_train, X_train_loc_feats, y_train, col_names = data_loader.load_data(train_mod, ref_std='expert_ref')
    n_feats = X_train.shape[-1]
    if test_mod.lower() in ['oc', 'ws']:  # This only directly makes sense for error quantification
        X_test, X_test_loc_feats, y_test, _ = data_loader.load_data(test_mod, ref_std='expert_ref')
        test_nans = np.isnan(X_test).any(axis=1)  # no nans will occur, just so that python doesn't complain
    else:  # Else just use an area to make a plot
        X_test, _, _ = data_loader.load_data(test_mod)
        X_test_loc_feats = np.copy(X_test)
        test_nans = np.isnan(X_test).any(axis=1)
        X_test = X_test[~test_nans]

        X_test = torch.tensor(X_test).float()
        X_test = X_test[:, -n_feats:]

    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).float()
    X_test = torch.tensor(X_test).float()

    # print(f"{X_train.shape=} {y_train.shape=} {X_test.shape=}")
    # print(f"{X_train=} {y_train=} {X_test=}")

    nu, num_steps, lr = set_hypers()
    kernels = {'Matern': ScaleKernel(MaternKernel(nu=nu, ard_num_dims=n_feats)),
               'RBF': ScaleKernel(RBFKernel(ard_num_dims=n_feats))}
    for k_name, kernel in kernels.items():
        likelihood = GaussianLikelihood()
        model = ExactGPModel(X_train, y_train, kernel, likelihood)

        losses, ls = train(X_train, y_train, model, likelihood, num_steps=num_steps, lr=lr)

        if plot_feature_importance:
            print('ARD lengthscale: ', ls)
            f_imp = [1 / x for x in ls]  # longer length scale means less important
            f_imp = np.array(f_imp) / np.sum(f_imp)  # normalize length scales
            plot.plot_f_importances(f_imp, col_names)

        y_preds = np.zeros((X_test_loc_feats.shape[0], 4))  # (:, lon, lat, y_pred, y_var)
        y_preds[:, 1] = X_test_loc_feats[:, 0]  # test y
        y_preds[:, 0] = X_test_loc_feats[:, 1]  # test x
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

        # print(X_test.shape, y_preds.shape, test_nans.shape)
        print(f"{f_mean=} {f_var=} {y_mean=}")
        # print(f"{y_pred.mean.shape=} {X_test.numpy().shape=} {y_preds.shape=}")
        with torch.no_grad():
            y_preds[~test_nans, 2] = y_mean.numpy()
            y_preds[~test_nans, 3] = f_var.numpy()

        X1 = y_preds[:, 0]
        X2 = y_preds[:, 1]

        plt.subplots()
        cmap = plot.set_colmap()
        bg = data_loader.load_bg(test_mod)
        # x0, x1, y0, y1 = y_preds[:, 0][0], y_preds[:, 0][test_h - 1], y_preds[:, 1][0], y_preds[:, 1][-1]
        plt.imshow(bg, extent=[np.min(X1),np.max(X1),np.min(X2),np.max(X2)], origin='upper')
        plt.gca().set_facecolor("tab:blue")
        plt.contourf(X1[:test_h], X2[::test_h], y_preds[:, 2].reshape((test_h, test_h)),
                     cmap=cmap, origin='upper', alpha=0.65)  # Plot the mean function as a filled contour
        plt.contour(X1[:test_h], X2[::test_h], y_preds[:, 3].reshape((test_h, test_h)),
                    levels=[1.96, 2.58], colors='black', linewidths=0.5, origin='upper', alpha=0.65)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        fig_url = 'C://Users/indy.dolmans/OneDrive - Nelen & Schuurmans/Pictures/maps/'
        name = fig_url + test_mod + '_' + train_mod + 'expert_' + k_name
        plt.savefig(name, bbox_inches='tight')
        plt.show()

        # losses = [loss.detach().numpy() for loss in losses]
        # plot.plot_loss(losses)
