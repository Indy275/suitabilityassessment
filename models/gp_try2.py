import configparser
import numpy as np

from data_util import load_data
from plotting import plot_gp

import torch
from torch.optim import Adam
from torch.nn import Parameter

import pyro

import pyro.contrib.gp as gp
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.distributions.util import eye_like


config = configparser.ConfigParser()
config.read('config.ini')

plot_data = int(config['PLOTTING']['plot_data'])
plot_pred = int(config['PLOTTING']['plot_pred'])


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


def run_model(train_mod):
    noise = config.getfloat('MODEL_PARAMS', 'noise')
    X_train, y_train = load_data.load_expert(train_mod)

    X = torch.tensor(X_train.transpose())
    y = torch.tensor(y_train)

    print("X shape: {} ; Y shape: {}".format(X.shape, y.shape))
    if plot_data:
        plot_gp.plot(X, y, plot_observed_data=True)

    Matern = gp.kernels.Matern32(input_dim=X.shape[1])
    RBF = gp.kernels.RBF(input_dim=X.shape[1])

    kernel_list = [RBF, Matern]
    for kernel in kernel_list:
        priors = [np.float64(1.01), np.float64(1.01)]
        kernel.set_prior("lengthscale", dist.Gamma(priors[0], priors[1]).to_event())
        kernel.set_prior("variance", dist.Gamma(priors[0]*7, priors[1]*6).to_event())
        gpr = gp.models.GPRegression(X, y, kernel, noise=torch.tensor(noise))
        if plot_pred:
            print("GP prior")
            plot_gp.plot(X, y, model=gpr, kernel=kernel, plot_predictions=True, plot_observed_data=True)

        N = X.size(0)
        f_loc = X.new_zeros(N)
        f_loc = Parameter(f_loc)
        zero_loc = X.new_zeros(f_loc.shape)
        identity = eye_like(X, N)
        likelihood = dist.MultivariateNormal(zero_loc, scale_tril=identity)
        guide = gp.models.VariationalGP(gpr.X, gpr.y, kernel, likelihood)

        # Define the optimizer and the inference algorithm
        optimizer = Adam(gpr.parameters(), lr=0.005)
        svi = SVI(gpr.model, guide.model, optimizer, loss=Trace_ELBO())

        # Perform training
        num_steps = 1000
        losses = []
        # train(gpr, num_steps)
        for step in range(num_steps):
            loss = svi.step()
            losses.append(loss)
        # OR:
        gp.util.train(gpr)

        X_new = torch.randn(10, 2)
        y_pred = gpr(X_new)
        if plot_pred:
            print("Posterior GP")
            plot_gp.plot_loss(losses)
            plot_gp.plot(X, y, model=gpr, plot_observed_data=True, plot_predictions=True)