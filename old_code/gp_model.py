import configparser
import numpy as np

from data_util import load_data
from plotting import gp_plot

import torch

import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist


config = configparser.ConfigParser()
config.read('config.ini')

plot_data = int(config['PLOTTING']['plot_data'])
plot_pred = int(config['PLOTTING']['plot_pred'])

Matern = gp.kernels.Matern32(input_dim=2)
RBF = gp.kernels.RBF(input_dim=2)

kernel_list = [RBF, Matern]


def train(gpr, num_steps=2000):
    optimizer = torch.optim.Adam(gpr.parameters(), lr=0.005)
    loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
    # hmc_kernel = pyro.infer.mcmc.HMC(gpr, step_size=0.0855, num_steps=4)
    # mcmc = pyro.infer.mcmc.MCMC(hmc_kernel, num_samples=500, warmup_steps=100)
    # mcmc.run(data)
    # mcmc.get_samples()['beta'].mean(0)
    losses = []
    variances = []
    lengthscales = []
    noises = []
    for i in range(num_steps):
        variances.append(gpr.kernel.variance.item())
        noises.append(gpr.noise.item())
        lengthscales.append(gpr.kernel.lengthscale.item())
        optimizer.zero_grad()
        loss = loss_fn(gpr.model, gpr.guide)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def run_model(train_mod, ref_std):
    noise = config.getfloat('MODEL_PARAMS', 'noise')
    X_train, y_train, X_train_orig, train_col_names = load_data.load_xy(train_mod, ref_std)
    print(X_train.shape, y_train.shape)
    # X_tr, y_tr = map(list, zip(*[(xi, yi) for xi, yi in zip(X_train, y_train) if yi > 0]))
    X_tr = X_train.reshape((40, 40, -1))
    y_tr = y_train.reshape((40, 40))
    X = torch.tensor(X_tr[:, :, 2])
    y = torch.tensor(y_tr)

    print("X shape: {} ; Y shape: {}".format(X.shape, y.shape))
    if plot_data:
        plot_gp.plot(X, y, plot_observed_data=True)  # let's plot the observed data

    for kernel in kernel_list:
        priors = [np.float64(1.01), np.float64(1.01)]
        kernel.set_prior("lengthscale", dist.Gamma(priors[0], priors[1]).to_event())
        kernel.set_prior("variance", dist.Gamma(priors[0]*7, priors[1]*6).to_event())
        gpr = gp.models.GPRegression(X, y, kernel, noise=torch.tensor(noise))
        if plot_pred:
            print("GP prior")
            plot_gp.plot(X, y, model=gpr, kernel=kernel, plot_predictions=True, plot_observed_data=True)

        train(gpr)

        if plot_pred:
            print("Posterior GP")
            # plot_gp.plot_loss(losses)
            plot_gp.plot(X, y, model=gpr, plot_observed_data=True, plot_predictions=True)
