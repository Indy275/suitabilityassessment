import configparser
import numpy as np

from data_util import load_data
from plotting import gp_plot

import torch
from torch.optim import Adam
from torch.nn import Parameter

import pyro

import pyro.contrib.gp as gp
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.distributions.util import eye_like
from pyro.nn import PyroSample

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


def run_model(train_mod, test_mod, test_w, test_h):
    noise = config.getfloat('MODEL_PARAMS', 'noise')
    X_train, y_train = load_data.load_expert(train_mod)
    # X_train = X_train[:, 2:7]
    X_train = X_train[:, :2] # Use only lat,lon
    X = torch.tensor(X_train)
    y = torch.tensor(y_train)

    X_test, _, _ = load_data.load_xy(test_mod)
    # X_test = X_test[:, 2:7]
    X_test = X_test[:, :2]  # Use only lat,lon
    X_test = torch.tensor(X_test).float()
    print(f"{X.shape=}{y.shape=}{X_test.shape=}")
    if plot_data:
        plot_gp.plot(X, y, X_test, test_w, test_h, plot_predictions=False)

    Matern = gp.kernels.Matern32(input_dim=X.shape[1])
    RBF = gp.kernels.RBF(input_dim=X.shape[1])
    Exp = gp.kernels.Exponential(input_dim=X.shape[1])
    kernel_list = [Exp, Matern, RBF]

    for kernel in kernel_list:
        priors = [np.float32(100.01), np.float32(100.01)]
        kernel.lengthscale = PyroSample(dist.Gamma(np.float32(100.), np.float32(5.)).to_event())
        kernel.variance = PyroSample(dist.Gamma(np.float32(5.) * 7, np.float32(10000.) * 6).to_event())
        print(f"{kernel.lengthscale=}{kernel.variance=}")

        gpr = gp.models.GPRegression(X, y, kernel, noise=torch.tensor(noise))
        if plot_pred:
            print("GP prior")
            plot_gp.plot(X, y, X_test, test_w, test_h, model=gpr, plot_predictions=False)
            plot_gp.plot_pred(X_test, test_w, test_h, model=gpr)

        # N = X.size(0)
        # f_loc = X.new_zeros(N)
        # f_loc = Parameter(f_loc)
        # zero_loc = X.new_zeros(f_loc.shape)
        # identity = eye_like(X, N)

        # likelihood = dist.MultivariateNormal(zero_loc, scale_tril=identity)
        # guide = gp.models.VariationalGP(gpr.X, gpr.y, kernel, likelihood)
        # guide = gp.models.VariationalGP(gpr.X, None, kernel, likelihood)

        # Define the optimizer and the inference algorithm
        # optimizer = Adam(gpr.parameters(), lr=0.005)
        # svi = SVI(gpr.model, guide.model, optimizer, loss=Trace_ELBO())

        # Perform training
        # num_steps = 1000
        # losses = []
        # train(gpr, num_steps)
        # for step in range(num_steps):
        #     loss = svi.step()
        #     losses.append(loss)
        # OR:
        # gp.util.train(gpr)

        # X_new = torch.randn(10, 2)
        # y_pred = gpr(X_new)
        if plot_pred:
            print("Posterior GP")
            plot_gp.plot_loss(losses)
            plot_gp.plot(X, y, X_test, test_w, test_h, model=gpr, plot_predictions=True)
            plot_gp.plot_pred(X_test, test_w, test_h, model=gpr)
