import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.interpolate import interp2d
from matplotlib.colors import LogNorm

import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist


def plot(X, y, plot_observed_data=False, plot_predictions=False, n_prior_samples=0, model=None, kernel=None,
         n_test=40, ax=None):
    # if ax is None:
    #     fig, ax = plt.subplots(figsize=(12, 6))
    X0 = X[:, 0].numpy()
    X1 = X[:, 1].numpy()
    if plot_observed_data:
        f = interp2d(X0, X1, y, kind="linear")

        x_coords = np.arange(min(X0) - 0.001, max(X0) + 0.001, 0.0005)
        y_coords = np.arange(min(X1) - 0.001, max(X1) + 0.001, 0.0005)
        Z = f(x_coords, y_coords)

        print(min(X0), max(X0))
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('auto')
        # ax.imshow(Z, extent=[min(X[0, :]), max(X[0, :]), min(X[1, :]), max(X[1, :])])
        ax.imshow(Z)
        # fig.axes.set_autoscale_on(False)
        # fig.axes.set_xlim(min(X[0, :]),max(X[0, :]))
        ax.set_xbound(min(X0), max(X0))
        ax.scatter(X0, X1, 200, marker='s', facecolors='none', edgecolors='r')
        ax.set_xbound(min(X0), max(X0))
        print(ax.get_xbound())
        print(ax.get_xlim())

    if plot_predictions:
        # X1test = torch.linspace(np.min(X.numpy()), np.max(X.numpy()), n_test).double()  # test inputs
        # X2test = torch.linspace(np.min(X.numpy()), np.max(X.numpy()), n_test).double()  # test inputs
        # xx, yy = torch.meshgrid([X1test, X2test])
        # print(xx, yy)
        # xx = np.repeat(X1test, len(X1test))
        X1test = torch.linspace(np.min(X0), np.max(X0), n_test).double()
        X2test = torch.linspace(np.min(X1), np.max(X1), n_test).double()
        print(X1test, X2test)
        xy = torch.vstack((X1test, X2test)).T

        print(xy.shape, X1test.shape, X2test.shape)
        # compute predictive mean and variance
        with torch.no_grad():
            if type(model) == gp.models.VariationalSparseGP:
                mean, cov = model(torch.Tensor(xy), full_cov=True)
            else:
                mean, cov = model(torch.Tensor(xy), full_cov=True, noiseless=False)
        sd = cov.diag().sqrt()  # standard deviation at each input point x
        print(mean.numpy())
        print(X1test.shape, X2test.shape, mean.shape)
        # ax.pcolormesh(X1test, X2test, mean.numpy(), "r", lw=2)  # plot the mean
        X, Y = np.meshgrid(range(xy.shape[0]), range(xy.shape[1]))
        plt.pcolormesh(X, Y, xy.T, shading='auto')
        plt.scatter(range(mean.numpy().shape[0]), np.zeros_like(mean.numpy()), c=mean.numpy(), cmap='viridis')
        # ax.fill_between(
        #     xy.numpy(),  # plot the two-sigma uncertainty about the mean
        #     (mean - 2.0 * sd).numpy(),
        #     (mean + 2.0 * sd).numpy(),
        #     color="C0",
        #     alpha=0.3,
        # )
    if n_prior_samples > 0:  # plot samples from the GP prior
        Xtest = torch.linspace(np.min(X.numpy()), np.max(X.numpy()), n_test).double()  # test inputs
        xy = torch.cartesian_prod(Xtest, Xtest)
        noise = (model.noise
                 if type(model) != gp.models.VariationalSparseGP
                 else model.likelihood.variance)
        cov = kernel.forward(xy) + noise.expand(n_test).diag()
        samples = dist.MultivariateNormal(
            torch.zeros(n_test), covariance_matrix=cov
        ).sample(sample_shape=(n_prior_samples,))
        ax.plot(xy.numpy(), samples.numpy().T, lw=2, alpha=0.4)

    ax.set_xlim(-1, 5.5)
    plt.show()


def plot_loss(loss):
    plt.plot(loss)
    plt.xlabel("Iterations")
    _ = plt.ylabel("Loss")
    plt.show()
