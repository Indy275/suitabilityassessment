import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern


def plot_prior_samples(gpr_model, n_samples, ax, label):
    """Plot samples drawn from the Gaussian process model.

    If the Gaussian process model is not trained then the drawn samples are
    drawn from the prior distribution. Otherwise, the samples are drawn from
    the posterior distribution. Be aware that a sample here corresponds to a
    function.

    Parameters
    ----------
    gpr_model : `GaussianProcessRegressor`
        A :class:`~sklearn.gaussian_process.GaussianProcessRegressor` model.
    n_samples : int
        The number of samples to draw from the Gaussian process distribution.
    ax : matplotlib axis
        The matplotlib axis where to plot the samples.
    """
    x = np.linspace(0, 5, 100)
    X = x.reshape(-1, 1)

    # y_mean, y_std = gpr_model.predict(X, return_std=True)
    y_samples = gpr_model.sample_y(X, n_samples)
    print(x.shape, y_samples.shape)
    for idx, single_prior in enumerate(y_samples.T):
        ax.plot(
            x,
            single_prior,
            linestyle="--",
            alpha=0.7,
            label=label,
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_ylim([-3, 3])


def plot_posterior_samples(gpr_model, n_samples, ax, label):
    """Plot samples drawn from the Gaussian process model.

    If the Gaussian process model is not trained then the drawn samples are
    drawn from the prior distribution. Otherwise, the samples are drawn from
    the posterior distribution. Be aware that a sample here corresponds to a
    function.

    Parameters
    ----------
    gpr_model : `GaussianProcessRegressor`
        A :class:`~sklearn.gaussian_process.GaussianProcessRegressor` model.
    n_samples : int
        The number of samples to draw from the Gaussian process distribution.
    ax : matplotlib axis
        The matplotlib axis where to plot the samples.
    """
    x = np.linspace(0, 5, 100)
    X = x.reshape(-1, 1)

    y_mean, y_std = gpr_model.predict(X, return_std=True)
    y_samples = gpr_model.sample_y(X, n_samples)

    # for idx, single_prior in enumerate(y_samples.T):
    #     ax.plot(
    #         x,
    #         single_prior,
    #         linestyle="--",
    #         alpha=0.7,
    #         label=label,
    #     )
    ax.plot(x, y_mean, label=label)
    ax.fill_between(
        x,
        y_mean - y_std,
        y_mean + y_std,
        alpha=0.1,
        # label=r"$\pm$ 1 std. dev.",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_ylim([-2, 2])


rbfkernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
gprrbf = GaussianProcessRegressor(kernel=rbfkernel, random_state=0)

maternkernel12 = 1.0 * Matern(nu=0.5, length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
gprmatern = GaussianProcessRegressor(kernel=maternkernel12, random_state=10)
maternkernel32 = 1.0 * Matern(nu=1.5, length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
gprmatern2 = GaussianProcessRegressor(kernel=maternkernel32, random_state=100)

if True:
    fig, axs = plt.subplots(nrows=1, sharey=True, figsize=(6,4))

    # plot prior
    plot_prior_samples(gprrbf, n_samples=1, ax=axs, label='Samples from RBF kernel')
    plot_prior_samples(gprmatern, n_samples=1, ax=axs, label=r'Sample from Matérn kernel ($\nu=1/2$)')
    plot_prior_samples(gprmatern2, n_samples=1, ax=axs, label=r'Sample from Matérn kernel ($\nu=3/2$)')

    axs.set_title("Samples from prior distribution")
    plt.legend()
    plt.tight_layout()
    plt.show()

fig, axs = plt.subplots(nrows=1, sharey=True, figsize=(6,4))

rng = np.random.RandomState(7)
X_train = rng.uniform(0, 5, 8).reshape(-1, 1)
y_train = np.sin((X_train[:, 0] - 2.5) ** 2)
gprrbf.fit(X_train, y_train)
gprmatern.fit(X_train, y_train)
gprmatern2.fit(X_train, y_train)
plot_posterior_samples(gprrbf, n_samples=1, ax=axs, label='Sample from RBF kernel')
plot_posterior_samples(gprmatern, n_samples=1, ax=axs, label=r'Sample from Matérn kernel ($\nu=1/2$)')
plot_posterior_samples(gprmatern2, n_samples=1, ax=axs, label=r'Sample from Matérn kernel ($\nu=3/2$)')
axs.scatter(X_train[:, 0], y_train, color="red", zorder=10, label="Observations")

axs.set_title("Samples from posterior distribution")
plt.legend()
plt.tight_layout()
plt.show()