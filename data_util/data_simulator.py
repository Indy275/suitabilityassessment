import numpy as np
import matplotlib.pyplot as plt
from models import GP_model
import torch
from sklearn.inspection import permutation_importance

from gpytorch.kernels import RBFKernel, ScaleKernel
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR


def calc_f_score(x):
    # x = 1 / np.log(x)
    x = -1 * x
    # x = 1 / x

    return x


def generate_expert_data(weights, n_samp=1000, noise_std=2):
    w0, w1, w2, w3, w4 = weights  # factor weights
    X = np.random.uniform(0, 5, size=(n_samp, 5))
    y = []
    for x in X:
        f0 = calc_f_score(x[0])
        f1 = calc_f_score(x[1])
        f2 = calc_f_score(x[2])
        f3 = calc_f_score(x[3])
        f4 = calc_f_score(x[4])

        noise = np.random.normal(0, noise_std)
        y_i = w0 * f0 + w1 * f1 + w2 * f2 + w3 * f3 + w4 * f4 + noise
        y.append(y_i)

    # plt.scatter(X[:, 0], y, marker='x')
    # plt.ylabel("Suitability score")
    # plt.xlabel(f"Feature {0} ({weights[0]} weighting) ({noise_std=})")
    # plt.show()
    return X, y


def fit_model(X_train, y_train, X_test, model):
    if model == 'gp':
        X_train_tensor = torch.tensor(X_train).float()
        y_train_tensor = torch.tensor(y_train).float()

        model, likelihood, losses, ls = GP_model.train(X_train_tensor, y_train_tensor,
                                                       ScaleKernel(RBFKernel(ard_num_dims=5)))

        f_imp = [1 / x for x in ls]  # longer length scale means less important
        f_imp = np.array(f_imp) / np.sum(f_imp)  # normalize length scales

        Lnglat = np.zeros((X_train.shape[0], 2))  # ignored
        nans = np.zeros(X_train.shape)  # ignored
        y_pred = GP_model.predict(Lnglat, X_test, nans, model, likelihood)
    elif model == 'gbr':
        model = GradientBoostingRegressor()
        model.fit(X_train, y_train)
        results = permutation_importance(model, X_train, y_train, n_repeats=10)
        f_imp = results.importances_mean
        f_imp = np.array(f_imp) / np.sum(f_imp)  # normalize
        y_pred = model.predict(X_test)

    elif model == 'svm':
        model = SVR(kernel='linear')
        model.fit(X_train, y_train)
        results = permutation_importance(model, X_train, y_train, n_repeats=10)
        f_imp = results.importances_mean
        f_imp = np.array(f_imp) / np.sum(f_imp)  # normalize
        y_pred = model.predict(X_test)

    return y_pred, f_imp


weights = [0.6, 0.4, 0, 0, 0]
print("true weights:", weights)
n_samples = [10, 50, 100, 1000]
n_samples = [50]
noise_vals = [0.01, 0.1, 1, 10]
model = 'svm'
crossval = 50  # number of iterations to obtain mean score

weight_errors = []
pred_errors = []
for nois in noise_vals:
    for samps in n_samples:
        weight_error = 0
        pred_error = 0
        for i in range(crossval):
            X_train, y_train = generate_expert_data(weights, n_samp=samps, noise_std=nois)
            X_test, y_test = generate_expert_data(weights, n_samp=samps, noise_std=nois)

            y_pred, f_imp = fit_model(X_train, y_train, X_test, model)
            # plot.plot_f_importances(f_imp, list('01234'))
            weight_error += mean_squared_error(weights, f_imp)
            pred_error += mean_squared_error(y_test, y_pred)
        weight_errors.append(weight_error / crossval)
        pred_errors.append(pred_error / crossval)

if len(n_samples) == 1:  # if constant number of samples, we modify the noise parameter
    plt.plot(list(range(len(noise_vals))), weight_errors)
    plt.xticks(list(range(len(noise_vals))), labels=[str(nv) for nv in noise_vals])
    plt.xlabel("Sample noise")
else:  # else, we modify the n_samples parameter
    plt.plot(list(range(len(n_samples))), weight_errors)
    plt.xticks(list(range(len(n_samples))), labels=[str(samp) for samp in n_samples])
    plt.xlabel("Number of samples")
plt.ylabel('MSE of learned weights')
plt.show()

if len(n_samples) == 1:  # if constant number of samples, we modify the noise parameter
    plt.plot(list(range(len(noise_vals))), pred_errors)
    plt.xticks(list(range(len(noise_vals))), labels=[str(nv) for nv in noise_vals])
    plt.xlabel("Sample noise")
else:  # else, we modify the n_samples parameter
    plt.plot(list(range(len(n_samples))), pred_errors)
    plt.xticks(list(range(len(n_samples))), labels=[str(samp) for samp in n_samples])
    plt.xlabel("Number of samples")
plt.ylabel('Prediction error')
plt.show()
