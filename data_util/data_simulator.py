import numpy as np
import matplotlib.pyplot as plt
from models import GP_model
import torch
from sklearn.inspection import permutation_importance

from gpytorch.kernels import RBFKernel, ScaleKernel
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


def generate_expert_data(n_samp=1000, noise_std=2, plot=False):
    X = np.random.uniform(0, 5, size=(n_samp, 1))
    rbfkernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
    gprrbf = GaussianProcessRegressor(kernel=rbfkernel, random_state=0)
    y_s = gprrbf.sample_y(X, 1)
    y = np.array([y_i + np.random.normal(0, noise_std) for y_i in y_s]).flatten()

    if plot:
        for feature in range(len(X[0])):
            plt.scatter(X[:, feature], y, marker='x')
            plt.ylabel("Suitability score")
            plt.show()
    return np.array(X), np.array(y)


def fit_model(X_train, y_train, X_test, model):
    if model == 'gp':
        X_train_tensor = torch.tensor(X_train).float()
        y_train_tensor = torch.tensor(y_train).float()

        model, likelihood, losses, ls = GP_model.train(X_train_tensor, y_train_tensor,
                                                       ScaleKernel(RBFKernel(ard_num_dims=5)), num_steps=100)

        Lnglat = np.zeros((X_test.shape[0], 2))  # ignored
        nans = np.zeros(X_test.shape[0], dtype=bool)  # ignored
        y_pred = GP_model.predict(Lnglat, torch.tensor(X_test).float(), nans, model, likelihood)
        y_pred = y_pred[:, 2]
    elif model == 'gbr':
        model = GradientBoostingRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    elif model == 'svm':
        model = SVR(kernel='linear')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    return y_pred


def run_simulation(noise_vals, n_samples, crossval, plot):
    gp_preds, gbr_preds = [], []
    for nois in noise_vals:
        for samps in n_samples:
            print(f"\nnoise: {nois} , n_samples: {samps}")
            gp_error, gbr_error = 0, 0
            for i in range(crossval):
                X_train, y_train = generate_expert_data(n_samp=samps, noise_std=nois, plot=plot)
                X_test, y_test = generate_expert_data(n_samp=200, noise_std=nois)

                gp_pred = fit_model(X_train, y_train, X_test, model='gp')
                gp_error += mean_squared_error(y_test, gp_pred)

                gbr_pred = fit_model(X_train, y_train, X_test, model='gbr')
                gbr_error += mean_squared_error(y_test, gbr_pred)

            gp_preds.append(gp_error / crossval)
            gbr_preds.append(gbr_error / crossval)

    return gp_preds, gbr_preds


n_samples = [10, 50, 100, 1000, 5000, 10000]
# n_samples = [5, 10, 50, 100, 500, 1000]
# n_samples = [50]
noise_vals = [0.001,0.01, 0.1, 1, 2]
noise_vals = [0.01]

crossval = 1  # number of iterations to obtain mean score
plot = True

gp_preds, gbr_preds = run_simulation(noise_vals, n_samples, crossval, plot)

fig_url = "C:/Users/indy.dolmans/OneDrive - Nelen & Schuurmans/Pictures/simulations/"

if len(n_samples) == 1:  # if constant number of samples, we modify the noise parameter
    plt.plot(noise_vals, gp_preds, label='GP', c='orange')
    plt.scatter(noise_vals, gp_preds, marker='x', c='orange')
    plt.plot(noise_vals, gbr_preds, label='GBR', c='b')
    plt.scatter(noise_vals, gbr_preds, marker='x', c='b')

    # plt.xticks(list(range(len(noise_vals))), labels=[str(nv) for nv in noise_vals])
    plt.xlabel("Sample noise")
    figname = 'MSE_Noise'
    plt.title(f'{n_samples=}, {crossval=}')
else:  # else, we modify the n_samples parameter
    plt.plot(n_samples, gp_preds, label='GP', c='orange')
    plt.scatter(n_samples, gp_preds, marker='x', c='orange')
    plt.plot(n_samples, gbr_preds, label='GBR', c='b')
    plt.scatter(n_samples, gbr_preds, marker='x', c='b')
    # plt.xticks(list(range(len(n_samples))), labels=[str(samp) for samp in n_samples])
    plt.xlabel("Number of samples")
    figname = 'MSE_nsamples'
    plt.title(f'{noise_vals=}, {crossval=}')
plt.legend()
plt.ylabel('Mean squared prediction error')
plt.savefig(fig_url+figname)

plt.show()
