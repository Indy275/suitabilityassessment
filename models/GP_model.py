import configparser
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from data_util import data_loader
from plotting import plot

import torch
from gpytorch.models import ExactGP
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel, LinearKernel, AdditiveKernel
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal

import matplotlib.pyplot as plt

config = configparser.ConfigParser()
config.read('config.ini')

data_url = config['DEFAULT']['data_url']
fig_url = config['DEFAULT']['fig_url']
plot_data = int(config['PLOTTING']['plot_data'])
plot_pred = int(config['PLOTTING']['plot_prediction'])
plot_feature_importance = int(config['PLOTTING']['plot_feature_importance'])
plot_variance = int(config['PLOTTING']['plot_feature_importance'])
digitize = int(config['PLOTTING']['digitize_prediction'])
sigmoidal_tf = int(config['PLOTTING']['sigmoid_prediction'])
coords_as_features = config.getboolean('DATA_SETTINGS', 'coords_as_features')


def save_imp(imp, names, model, modifier):
    impFrame = pd.DataFrame(data=dict({'feature': names, 'importance': imp}))
    impFrame.to_csv(data_url + '/' + modifier + "/GP_" + model + "_fimp.csv", index=False)


def set_hypers():
    nu_coords = float(config['MODEL_PARAMS_GP']['nu_coords'])
    nu_feats = float(config['MODEL_PARAMS_GP']['nu_feats'])
    num_steps = int(config['MODEL_PARAMS_GP']['num_steps'])
    lr = float(config['MODEL_PARAMS_GP']['lr'])
    lasso = float(config['MODEL_PARAMS_GP']['lasso'])
    return nu_coords, nu_feats, num_steps, lr, lasso


class ExactGPModel(ExactGP):
    def __init__(self, X_train, y_train, kernel1, likelihood, lasso_weight):
        super(ExactGPModel, self).__init__(X_train, y_train, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = kernel1
        self.lasso_weight = lasso_weight

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def get_regularization_penalty(self):
        return self.lasso_weight * \
            (torch.norm(self.covar_module.base_kernel.lengthscale, 1))


class LngLatExactGPModel(ExactGP):
    def __init__(self, X_train, y_train, kernel1, kernel2, likelihood, lasso_weight=1e-5):
        super(LngLatExactGPModel, self).__init__(X_train, y_train, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module_coord = kernel1
        self.covar_module_feat = kernel2
        self.covar_module = AdditiveKernel(self.covar_module_coord, self.covar_module_feat)
        self.lasso_weight = lasso_weight

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def get_regularization_penalty(self):
        return self.lasso_weight * \
            (torch.norm(self.covar_module_coord.base_kernel.lengthscale, 1) +
             torch.norm(self.covar_module_feat.base_kernel.lengthscale, 1))


def trainlnglat(train_x, train_y, kernel1, kernel2, num_steps=500, lr=0.01, lasso=1e-4):
    likelihood = GaussianLikelihood()
    model = LngLatExactGPModel(train_x, train_y, kernel1, kernel2, likelihood, lasso_weight=lasso)

    # Set trainable
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = ExactMarginalLogLikelihood(likelihood, model)

    losses = []
    for i in range(num_steps):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y) + model.get_regularization_penalty()
        loss.backward()
        losses.append(loss)
        if i % 50 == 0:
            print('Iter %d/%d - Loss: %.3f   noise: %.3f' % (
                i + 1, num_steps, loss.item(),
                model.likelihood.noise.item()
            ))
        optimizer.step()

    print('Training completed - Loss: %.3f   noise: %.3f   scales: %.3f , %.3f   SNR: %.3f' % (
        losses[-1].item(),
        model.likelihood.noise.item(),
        model.covar_module_coord.outputscale.item(),
        model.covar_module_feat.outputscale.item(),
        model.covar_module_feat.outputscale.item() / model.likelihood.noise.item()
    ))

    if not isinstance(kernel2, LinearKernel):
        with torch.no_grad():
            ls_coord = model.covar_module_coord.base_kernel.lengthscale.detach().numpy()
            ls_feat = model.covar_module_feat.base_kernel.lengthscale.detach().numpy()
        ls = [y for x in [ls_coord[0], ls_feat[0]] for y in x]
    else:
        ls = []

    return model, likelihood, losses, ls


def train(train_x, train_y, kernel1, num_steps=500, lr=0.01, lasso=1e-4):
    likelihood = GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, kernel1, likelihood, lasso_weight=lasso)
    # Set trainable
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = ExactMarginalLogLikelihood(likelihood, model)

    losses = []
    for i in range(num_steps):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y) + model.get_regularization_penalty()
        loss.backward()
        losses.append(loss)
        if i % 50 == 0:
            print('Iter %d/%d - Loss: %.3f   noise: %.3f' % (
                i + 1, num_steps, loss.item(),
                model.likelihood.noise.item()
            ))
        optimizer.step()

    print('Training completed - Loss: %.3f   noise: %.3f   scales: %.3f    SNR: %.3f' % (
        losses[-1].item(),
        model.likelihood.noise.item(),
        model.covar_module.outputscale.item(),
        model.covar_module.outputscale.item() / model.likelihood.noise.item()
    ))

    if not isinstance(kernel1, LinearKernel):
        with torch.no_grad():
            ls = model.covar_module.base_kernel.lengthscale.detach().numpy()[0]
    else:
        ls = []

    return model, likelihood, losses, ls


def predict(LngLat, X_test, test_nans, model, likelihood):
    y_preds = np.zeros((test_nans.shape[0], 4))  # (:, lon, lat, y_pred, y_var)
    y_preds[:, 0] = LngLat[:, 0]  # test longitude
    y_preds[:, 1] = LngLat[:, 1]  # test latitude
    y_preds[test_nans, 2] = np.nan  # mean
    y_preds[test_nans, 3] = np.nan  # var

    X_test = X_test[~test_nans]

    # test phase
    model.eval()
    likelihood.eval()
    f_preds = model(X_test)
    y_pred = likelihood(model(X_test))

    f_mean = f_preds.mean
    f_var = f_preds.variance
    # f_covar = f_preds.covariance_matrix
    y_mean = y_pred.mean

    with torch.no_grad():
        y_preds[~test_nans, 2] = y_mean.numpy()
        y_preds[~test_nans, 3] = f_var.numpy()

    return y_preds


def evaluate(test_mod, model, likelihood, test_indices):
    eval_loader = data_loader.DataLoader(test_mod, ref_std='expert_ref')
    x_eval, y_eval, eval_lnglat, test_col_names = eval_loader.preprocess_input()
    test_nans = np.isnan(x_eval).any(axis=1)
    x_eval = x_eval[:, test_indices]

    x_eval = torch.tensor(x_eval).float()
    y_pred = predict(eval_lnglat, x_eval, test_nans, model, likelihood)
    mse = mean_squared_error(y_eval, y_pred[:, -2])
    print("Test MSE: {:.4f}".format(mse))


def plot_feature(data_loader, model, likelihood, ind):
    x_train, y_train, lnglat, col_names = data_loader.preprocess_input()
    orig_df = np.array(data_loader.load_orig_df())[:, ind]
    x_train = x_train[:, ind]
    col_names = np.array(col_names)[ind]

    n_samples = 200

    for feature in range(len(ind)):
        nullvals = np.min(x_train, axis=0)
        x_eval = np.tile(nullvals, (n_samples, 1))

        minval = np.min(x_train[:, feature])
        maxval = np.max(x_train[:, feature])
        x_feature = np.linspace(minval, maxval, n_samples)

        minval_orig = np.min(orig_df[:, feature])
        maxval_orig = np.max(orig_df[:, feature])
        x_orig = np.linspace(minval_orig, maxval_orig, n_samples)

        x_eval[:, feature] = x_feature
        x_eval = torch.tensor(x_eval).float()
        test_nans = np.array([False] * n_samples)
        lnglat = np.zeros((n_samples, 2))
        y_pred = predict(lnglat, x_eval, test_nans.T, model, likelihood)
        plt.scatter(orig_df[:, feature], y_train, marker='x')
        plt.plot(x_orig, y_pred[:, 2], color='orange')
        plt.fill_between(x_orig, y_pred[:, 2] - y_pred[:, 3], y_pred[:, 2] + y_pred[:, 3], alpha=0.5)
        plt.xlabel(col_names[feature])
        plt.ylabel("Suitability score")
        plt.show()


def run_model(train_mod, test_mod):
    x_subset = True
    train_data = data_loader.DataLoader(train_mod, ref_std='expert_ref')
    test_data = data_loader.DataLoader(test_mod, ref_std='testdata')

    X_train, y_train, train_lnglat, train_col_names = train_data.preprocess_input()
    X_test, test_nans, test_lnglat, test_size, test_col_names = test_data.preprocess_input()

    bg_test = test_data.load_bg()
    assert train_col_names == test_col_names

    if plot_data:
        bg_train = train_data.load_bg()
        plot.plot_y(train_data, bg_train, ref_std='expert_ref')

    x_indices = list(range(X_train.shape[1]))
    if x_subset:
        x_indices = [0, 1, 3]
        X_train = X_train[:, x_indices]
        X_test = X_test[:, x_indices]
        train_col_names = np.array(train_col_names)[x_indices]

    print(f'Training model. Train shape: {X_train.shape}. Test shape: {X_test.shape}')
    print(f'Train variables: {train_col_names} at {x_indices=}')

    X_train_tensor = torch.tensor(X_train).float()
    y_train_tensor = torch.tensor(y_train).float()
    X_test_tensor = torch.tensor(X_test).float()

    coord_nu, feat_nu, num_steps, lr, lasso = set_hypers()

    if coords_as_features:
        feature_dims = tuple(list(range(len(x_indices)))[2:])
        coord_kernel = ScaleKernel(MaternKernel(nu=coord_nu, ard_num_dims=2, active_dims=(0, 1)))
        kernels = {'RBF': ScaleKernel(RBFKernel(ard_num_dims=len(x_indices[2:]), active_dims=feature_dims)),
                   'Matern': ScaleKernel(MaternKernel(nu=feat_nu, ard_num_dims=len(x_indices[2:]), active_dims=feature_dims))}
    else:
        kernels = {'RBF': ScaleKernel(RBFKernel(ard_num_dims=len(x_indices), active_dims=tuple(range(len(x_indices))))),
                   'Matern': ScaleKernel(MaternKernel(nu=feat_nu, ard_num_dims=len(x_indices), active_dims=tuple(range(len(x_indices)))))}

    for k_name, feature_kernel in kernels.items():
        if coords_as_features:
            model, likelihood, losses, ls = trainlnglat(X_train_tensor, y_train_tensor, coord_kernel, feature_kernel,
                                                  num_steps=num_steps, lr=lr, lasso=lasso)
        else:
            model, likelihood, losses, ls = train(X_train_tensor, y_train_tensor, feature_kernel,
                                                  num_steps=num_steps, lr=lr, lasso=lasso)
        y_preds = predict(test_lnglat, X_test_tensor, test_nans, model, likelihood)

        if digitize:
            n_quantiles = 11
            quantiles = np.linspace(0, 1, n_quantiles)
            y_preds[~test_nans, 2] = np.digitize(y_preds[~test_nans, 2],
                                                 np.nanquantile(y_preds[~test_nans, 2], quantiles))
        elif sigmoidal_tf:
            print("preds were in range [{0:.3f},{1:.3f}]".format(np.nanmin(y_preds[:, 2]), np.nanmax(y_preds[:, 2])))
            sigmoid = lambda x: 1 / (1 + np.exp(-.5 * (x - .5)))
            y_preds[~test_nans, 2] = sigmoid(y_preds[~test_nans, 2])
            print("preds now in range [{0:.3f},{1:.3f}]".format(np.nanmin(y_preds[:, 2]), np.nanmax(y_preds[:, 2])))

        if plot_pred:
            fig_name = fig_url + '/' + test_mod + '_' + train_mod + '_GP' + k_name
            fig_title = f'GP-{k_name} kernel predictions trained on {train_mod} labels'
            if train_mod == test_mod:  # plot train labels on top of prediction
                train_labs = np.column_stack([train_lnglat[:, 0], train_lnglat[:, 1], y_train])
                plot.plot_prediction(y_preds, test_size, fig_name, title=fig_title, train_labs=train_labs, contour=True, bg=bg_test,
                                     savefig=True)
            else:
                plot.plot_prediction(y_preds, test_size, fig_name, title=fig_title, contour=True, bg=bg_test, savefig=True)

        if plot_variance:
            fig_title = f'GP-{k_name} kernel variance trained on {train_mod} labels'
            y_preds[:, 2] = y_preds[:, 3] * -1  # Swap to plot variance; negated because high variance is worse
            if train_mod == test_mod:  # plot train labels on top of prediction
                train_labs = np.column_stack([train_lnglat[:, 0], train_lnglat[:, 1], y_train])
                plot.plot_prediction(y_preds, test_size, title=fig_title, train_labs=train_labs, contour=True,
                                     bg=bg_test, savefig=False)
            else:
                plot.plot_prediction(y_preds, test_size, title=fig_title, contour=True, bg=bg_test, savefig=False)

        if plot_feature_importance and not k_name == 'Linear':
            print('ARD lengthscale: ', ls)
            f_imp = [1 / x for x in ls]  # longer length scale means less important
            f_imp = np.array(f_imp) / np.sum(f_imp)  # normalize length scales
            plot.plot_f_importances(f_imp, train_col_names)
            save_imp(f_imp, train_col_names, k_name, test_mod)

        if test_mod.startswith(('ws', 'oc')):  # test labels exist only for these data sets
            evaluate(test_mod, model, likelihood, x_indices)

        plot_feature(train_data, model, likelihood, x_indices)