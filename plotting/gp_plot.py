import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.interpolate import interp2d, LinearNDInterpolator, griddata
from matplotlib.colors import LogNorm

import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist


def plot(X, y, Xtest, test_w, test_h, model=None, plot_predictions=False, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    X0 = X[:, 0].numpy()
    X1 = X[:, 1].numpy()
    # f = interp2d(X0, X1, y, kind="linear")  # NEVER use this function: its buggy as hell (confirmed on SO)

    # x_coords = np.arange(min(X0) - 0.001, max(X0) + 0.001, 0.0005)
    # y_coords = np.arange(min(X1) - 0.001, max(X1) + 0.001, 0.0005)
    # x_coords = np.arange(min(X0) - 0.001, max(X0) + 0.001, 0.01)
    # y_coords = np.arange(min(X1) - 0.001, max(X1) + 0.001, 0.01)

    # Z = f(x_coords, y_coords)
    # ax.pcolormesh(x_coords, y_coords, Z, shading='auto')
    # ax.imshow(Z, extent=[min(x_coords), max(x_coords), min(y_coords), max(y_coords)])

    # ax.scatter(X0, X1, 100, marker='s', facecolors='none', edgecolors='r')
    ax.scatter(X0, X1, c=y, cmap='viridis')
    ax.set_xlim(min(X0), max(X0))
    ax.set_ylim(min(X1), max(X1))

    if plot_predictions:
        plot_pred(Xtest, test_w, test_h, model=model, ax=ax)
    # if n_prior_samples > 0:  # plot samples from the GP prior
    #     Xtest = torch.linspace(np.min(X.numpy()), np.max(X.numpy()), n_test).double()  # test inputs
    #     xy = torch.cartesian_prod(Xtest, Xtest)
    #     noise = (model.noise
    #              if type(model) != gp.models.VariationalSparseGP
    #              else model.likelihood.variance)
    #     cov = kernel.forward(xy) + noise.expand(n_test).diag()
    #     samples = dist.MultivariateNormal(
    #         torch.zeros(n_test), covariance_matrix=cov
    #     ).sample(sample_shape=(n_prior_samples,))
    #     ax.plot(xy.numpy(), samples.numpy().T, lw=2, alpha=0.4)

    plt.show()


def plot_pred(Xtest, test_w, test_h, model=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    X0 = Xtest[:, 0].numpy()
    X1 = Xtest[:, 1].numpy()
    # f = interp2d(X0, X1, y, kind="linear")  # NEVER use this function: its buggy as hell (confirmed on SO)

    # x_coords = np.arange(min(X0) - 0.001, max(X0) + 0.001, 0.0005)
    # y_coords = np.arange(min(X1) - 0.001, max(X1) + 0.001, 0.0005)
    # x_coords = np.arange(min(X0) - 0.001, max(X0) + 0.001, 0.01)
    # y_coords = np.arange(min(X1) - 0.001, max(X1) + 0.001, 0.01)

    # Z = f(x_coords, y_coords)
    # ax.pcolormesh(x_coords, y_coords, Z, shading='auto')
    # ax.imshow(Z, extent=[min(x_coords), max(x_coords), min(y_coords), max(y_coords)])

    # ax.scatter(X0, X1, 100, marker='s', facecolors='none', edgecolors='r')
    ax.set_xlim(min(X0), max(X0))
    ax.set_ylim(min(X1), max(X1))

    # X1test = torch.linspace(np.min(X.numpy()), np.max(X.numpy()), n_test).double()  # test inputs
    # X2test = torch.linspace(np.min(X.numpy()), np.max(X.numpy()), n_test).double()  # test inputs
    # xx, yy = torch.meshgrid([X1test, X2test])
    # print(xx, yy)
    # xx = np.repeat(X1test, len(X1test))
    print(f"{Xtest.shape=}")
    # X1test = torch.linspace(np.min(Xtest[:,0]), np.max(Xtest[:,0]), n_test).float()
    # X2test = torch.linspace(np.min(Xtest[:,1]), np.max(Xtest[:,1]), n_test).float()

    # grid_x, grid_y = np.meshgrid(X1test, X2test)
    # xy = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    # compute predictive mean and variance
    with torch.no_grad():
        if type(model) == gp.models.VariationalSparseGP:
            mean, cov = model(Xtest, full_cov=True)
        else:
            mean, cov = model(Xtest, full_cov=True, noiseless=False)
    sd = cov.diag().sqrt()  # standard deviation at each input point x
    print(f"{Xtest.shape=} {mean.shape=}")
    # X, Y = np.meshgrid(range(xy.shape[0]), range(xy.shape[1]))
    # plt.pcolormesh(X, Y, xy.T, shading='auto')
    # plt.scatter(range(mean.numpy().shape[0]), np.zeros_like(mean.numpy()), c=mean.numpy(), cmap='viridis')
    mean = np.reshape(mean, (test_w, test_h))
    # ax.pcolormesh(X1test, X2test, mean, shading='auto')
    ax.imshow(mean, extent=[min(Xtest[:,0]), max(Xtest[:,0]), min(Xtest[:,1]), max(Xtest[:,1])])

    plt.show()


def plot_loss(loss):
    plt.plot(loss)
    plt.xlabel("Iterations")
    _ = plt.ylabel("Loss")
    plt.show()
