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

config = configparser.ConfigParser()
config.read('config.ini')

data_url = config['DEFAULT']['data_url']
plot_data = int(config['PLOTTING']['plot_data'])
plot_pred = int(config['PLOTTING']['plot_pred'])
plot_feature_importance = int(config['PLOTTING']['plot_feature_importance'])


def save_imp(imp, names, model, modifier):
    impFrame = pd.DataFrame(data=dict({'feature': names, 'importance': imp}))
    impFrame.to_csv(data_url + '/' + modifier + "/GP_" + model + "_fimp.csv", index=False)


def set_hypers():
    nu_coords = float(config['MODEL_PARAMS_GP']['nu_coords'])
    nu_feats = float(config['MODEL_PARAMS_GP']['nu_feats'])
    num_steps = int(config['MODEL_PARAMS_GP']['num_steps'])
    lr = float(config['MODEL_PARAMS_GP']['lr'])
    return nu_coords, nu_feats, num_steps, lr


class ExactGPModel(ExactGP):
    def __init__(self, X_train, y_train, kernel1, kernel2, likelihood, lasso_weight=1e-4):
        super(ExactGPModel, self).__init__(X_train, y_train, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module_coord = kernel1
        self.covar_module_feat = kernel2
        self.covar_module = AdditiveKernel(self.covar_module_coord, self.covar_module_feat)
        # self.covar_module = ScaleKernel(AdditiveKernel(self.covar_module_feat, num_dims=X_train.shape[-1]))
        self.lasso_weight = lasso_weight

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def get_regularization_penalty(self):
        return self.lasso_weight * \
            (torch.norm(self.covar_module_coord.base_kernel.lengthscale, 1) +
             torch.norm(self.covar_module_feat.base_kernel.lengthscale, 1))


def preprocess_input(X_train, y_train, X_test, col_names):
    lonlat = 0  # 0 if include lonlat; 2 if exclude
    testLngLat = X_test[:, :2]

    X_train = X_train[y_train != 0]
    X_train = X_train[:, lonlat:]
    y_train = y_train[y_train != 0]

    test_nans = np.isnan(X_test).any(axis=1)
    X_test = X_test[~test_nans]
    X_test_feats = X_test[:, lonlat:]
    col_names = col_names[lonlat:-1]
    return X_train, y_train, X_test_feats, testLngLat, test_nans, col_names


def train(train_x, train_y, kernel1, kernel2, num_steps=500, lr=0.01):
    likelihood = GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, kernel1, kernel2, likelihood)

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


def predict(LngLat, X_test, test_nans, model, likelihood, test_mod):
    y_preds = np.zeros((test_nans.shape[0], 4))  # (:, lon, lat, y_pred, y_var)
    y_preds[:, 0] = LngLat[:, 0]  # test y
    y_preds[:, 1] = LngLat[:, 1]  # test x
    y_preds[test_nans, 2] = np.nan
    y_preds[test_nans, 3] = np.nan

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

    pd.DataFrame(y_preds).to_csv(data_url + '/' + test_mod + "/GP_pred.csv")
    return y_preds


def evaluate(test_mod, model, likelihood):
    eval_loader = data_loader.DataLoader(test_mod, ref_std='expert_ref')
    x_eval, y_eval, eval_lnglat, test_col_names = eval_loader.preprocess_input()
    test_nans = np.isnan(x_eval).any(axis=1)

    x_eval = torch.tensor(x_eval).float()
    y_pred = predict(eval_lnglat, x_eval, test_nans, model, likelihood, test_mod)
    mse = mean_squared_error(y_eval, y_pred[:, -2])
    print("Test MSE: {:.4f}".format(mse))


def run_model(train_mod, test_mod):
    train_data = data_loader.DataLoader(train_mod, ref_std='expert_ref')
    test_data = data_loader.DataLoader(test_mod, ref_std='testdata')

    X_train, y_train, train_lnglat, train_col_names = train_data.preprocess_input()
    X_test, test_nans, test_lnglat, test_size, test_col_names = test_data.preprocess_input()

    bg_test = test_data.load_bg()
    assert train_col_names == test_col_names

    if plot_data:
        bg_train = train_data.load_bg()
        plot.plot_y(train_data, bg_train, ref_std='expert_ref')

    # ind = 6
    # ind = [0, 1]  # Longitude, Latitude
    # X_train = X_train[:, ind].reshape(-1, 1)  # temp
    # X_test = X_test[:, ind].reshape(-1, 1)  # temp
    # train_col_names = train_col_names[ind]  # temp    X_train = torch.tensor(X_train).float()

    print(f'Training model. Train shape: {X_train.shape}. Test shape: {X_test.shape}')
    print(f'Train variables: {train_col_names}')

    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).float()
    X_test = torch.tensor(X_test).float()

    coord_nu,feat_nu, num_steps, lr = set_hypers()
    coord_kernel = ScaleKernel(MaternKernel(nu=coord_nu, ard_num_dims=2, active_dims=(0, 1)))
    kernels = {'RBF': ScaleKernel(RBFKernel(ard_num_dims=5, active_dims=(2, 3, 4, 5, 6))),}
               # 'Linear': ScaleKernel(LinearKernel(active_dims=(2, 3, 4, 5, 6))),
               # 'Matern': ScaleKernel(MaternKernel(nu=feat_nu, ard_num_dims=5, active_dims=(2, 3, 4, 5, 6)))}
    # kernels = {'Matern': ScaleKernel(RBFKernel())}  # msk
    for k_name, feature_kernel in kernels.items():
        model, likelihood, losses, ls = train(X_train, y_train, coord_kernel, feature_kernel, num_steps=num_steps,
                                              lr=lr)
        y_preds = predict(test_lnglat, X_test, test_nans, model, likelihood, test_mod)
        y_preds[~test_nans, 2] = np.digitize(y_preds[~test_nans, 2], np.nanquantile(y_preds[~test_nans, 2], [0, 0.2, 0.4, 0.6, 0.8, 1.0]))
        # y_preds[:, 2] = y_preds[:, 3] * -1  # Swap to plot variance; negated because high variance is worse
        print(y_preds)
        if plot_pred:
            fig_url = 'C://Users/indy.dolmans/OneDrive - Nelen & Schuurmans/Pictures/maps/'
            fig_name = fig_url + test_mod + '_' + train_mod + '_GP' + k_name

            if train_mod == test_mod:  # plot train labels on top of prediction
                train_labs = np.column_stack([train_lnglat[:, 0], train_lnglat[:, 1], y_train])
                plot.plot_prediction(y_preds, test_size, fig_name, train_labs=train_labs, contour=True, bg=bg_test,
                                     savefig=True)
            else:
                plot.plot_prediction(y_preds, test_size, fig_name, contour=True, bg=bg_test, savefig=True)

            losses = [loss.detach().numpy() for loss in losses]
            plot.plot_loss(losses)

        if plot_feature_importance and not k_name == 'Linear':
            print('ARD lengthscale: ', ls)
            f_imp = [1 / x for x in ls]  # longer length scale means less important
            f_imp = np.array(f_imp) / np.sum(f_imp)  # normalize length scales
            plot.plot_f_importances(f_imp, train_col_names)
            save_imp(f_imp, train_col_names, k_name, test_mod)

        if test_mod.startswith(('ws', 'oc')):  # test labels exist only for these data sets
            evaluate(test_mod, model, likelihood)
