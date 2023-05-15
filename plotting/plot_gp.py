import matplotlib.pyplot as plt
import torch
import numpy as np

import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist


def plot(X, y, plot_observed_data=False, plot_predictions=False, n_prior_samples=0, model=None, kernel=None,
         n_test=40, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    if plot_observed_data:
        ax.plot(X.numpy(), y.numpy(), "kx")
    if plot_predictions:
        X1test = torch.linspace(np.min(X.numpy()), np.max(X.numpy()), n_test).double()  # test inputs
        X2test = torch.linspace(np.min(X.numpy()), np.max(X.numpy()), n_test).double()  # test inputs
        xx, yy = torch.meshgrid([X1test, X2test])
        print(xx, yy)
        # xx = np.repeat(X1test, len(X1test))
        # print(xx.shape)
        # compute predictive mean and variance
        with torch.no_grad():
            if type(model) == gp.models.VariationalSparseGP:
                mean, cov = model(torch.Tensor(xx), full_cov=True)
            else:
                mean, cov = model(torch.Tensor(xx), full_cov=True, noiseless=False)
        # sd = cov.diag().sqrt()  # standard deviation at each input point x
        print(mean.numpy())
        print(X1test.shape, X2test.shape, mean.shape)
        ax.pcolormesh([X1test, X2test], mean.numpy(), "r", lw=2)  # plot the mean
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
