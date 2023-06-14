import configparser
import numpy as np

from data_util import load_data
from plotting import plot

import matplotlib.pyplot as plt

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


class ExactGPModel(ExactGP):
    def __init__(self, X_train, y_train, kernel, likelihood):
        super(ExactGPModel, self).__init__(X_train, y_train, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


def train(train_x, train_y, model, likelihood, num_steps=500):
    # Set trainable
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = ExactMarginalLogLikelihood(likelihood, model)

    losses = []
    for i in range(num_steps):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
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
        print('ARD lengthscale: ',ls[0])
    return losses


def transform_expert(mod):
    X, y = load_data.load_expert(mod)
    X_feats = X[:, 2:7]
    # X_feats = X_feats[:, :2] # Use only lat,lon
    X_loc_feats = X
    X = torch.tensor(X_feats).float()
    y = torch.tensor(y).float()
    return X, X_loc_feats, y


def run_model(train_mod, test_mod, test_w, test_h):
    X_train, X_train_loc_feats, y_train = transform_expert(train_mod)
    n_feats = X_train.shape[-1]
    if test_mod.lower() in ['oc', 'ws']:
        X_test, X_test_loc_feats, y_test = transform_expert(test_mod)
    else:
        X_test, _, _ = load_data.load_xy(test_mod)
        X_test = torch.tensor(X_test).float()
        X_test = X_test[:, -n_feats:]
    # X_test = X_test[2500:5000, :]
    test_nans = np.isnan(X_test).any(axis=1)
    print(sum(test_nans))
    X_test = X_test[~test_nans]

    print(f"{X_train.shape=} {y_train.shape=} {X_test.shape=}")

    kernels = [ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=n_feats)), ScaleKernel(RBFKernel(ard_num_dims=n_feats))]

    for kernel in kernels:
        likelihood = GaussianLikelihood()
        model = ExactGPModel(X_train, y_train, kernel, likelihood)

        losses = train(X_train, y_train, model, likelihood, num_steps=500)

        y_preds = np.zeros(X_test.shape[0])
        y_preds[test_nans] = np.nan
        # test phase
        model.eval()
        likelihood.eval()
        f_preds = model(X_test)
        y_pred = likelihood(model(X_test))

        f_mean = f_preds.mean
        f_var = f_preds.variance
        # f_covar = y_preds.covariance_matrix
        y_mean = y_pred.mean

        print(f"{f_mean=} {f_var=} {y_mean=}")
        print(f"{y_pred.mean.shape=} {X_test.numpy().shape=} {X_train.numpy().shape=}")
        with torch.no_grad():
            y_pred = y_pred.mean.numpy()
            y_preds[~test_nans] = y_pred
            height = test_h
            plot.plot_prediction(y_preds, height)
            # plot.plot_prediction(f_var, height)
        losses = [loss.detach().numpy() for loss in losses]
        plot.plot_loss(losses)
