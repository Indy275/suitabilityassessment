import numpy as np
import matplotlib.pyplot as plt
from models.OCGPR import OCGP
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern

from sklearn.metrics import mean_squared_error, accuracy_score, auc, roc_auc_score
from sklearn.svm import OneClassSVM


def generate_expert_data(n_samp=1000, noise_std=2, plot=False):
    X = np.random.uniform(0, 5, size=(n_samp, 5))
    pos_threshold = 0
    rbfkernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
    gprrbf = GaussianProcessRegressor(kernel=rbfkernel, random_state=0)
    y_s = gprrbf.sample_y(X, 1)
    y = np.array([y_i + np.random.normal(0, noise_std) for y_i in y_s])

    y_bin = []
    for y_i in y:
        if y_i > pos_threshold:
            y_bin.append(1)
        else:
            y_bin.append(0)

    if plot:
        for feature in range(len(X[0])):
            x = np.linspace(min(X[:, feature]), max(X[:, feature]), 100)
            print(x.shape, y_s.shape, y.shape)

            # plt.plot(x,y_s[feature],linestyle="--")
            plt.scatter(X[:, feature], y, marker='x')
            plt.hlines(pos_threshold, xmin=min(X[:, feature]), xmax=max(X[:, feature]))
            plt.ylabel("Suitability score")
            plt.show()
    return np.array(X), np.array(y_bin)


def fit_model(X_train, X_test, model):
    if model == 'gp':
        ocgp = OCGP()
        ocgp.seKernel(X_train, X_test, 2, 0.0045)
        modes = ['mean', 'var', 'pred', 'ratio']
        y_preds = []
        for i in range(len(modes)):
            y_pred = np.squeeze(np.array(ocgp.getGPRscore(modes[i])))
            y_preds.append(y_pred)

    elif model == 'svm':
        kernel = 'linear'  # 'linear'  'rbf'
        predictor = OneClassSVM(kernel=kernel)
        predictor.fit(X=X_train)
        y_preds = predictor.score_samples(X_test)
        # y_preds = predictor.predict(X_test)

    return y_preds


def ystats(y):
    print("min:", min(y), "max:", max(y), "mean:", np.mean(y))


def binary_accuracy(y_test, y):
    # ystats(y)
    pos_threshold = 1
    y_pred = [1 if y_i >= pos_threshold else 0 for y_i in y]
    return accuracy_score(y_test, y_pred)


def auc_score(y_test, y):
    # y_test, y = zip(*sorted(zip(y_test, y)))
    y_test, y = zip(*sorted(zip(y_test, y)))
    return roc_auc_score(y_test, y)


def run_simulation(acc_dict, auc_dict, noise_vals, n_samples, crossval, plot, model):
    preds_0, preds_1, preds_2, preds_3, preds_4 = [], [], [], [], []
    aucs_0, aucs_1, aucs_2, aucs_3, aucs_4 = [], [], [], [], []
    for nois in noise_vals:
        for samps in n_samples:
            print(f"\nnoise: {nois} , n_samples: {samps}")
            pred_0, pred_1, pred_2, pred_3, pred_4 = 0, 0, 0, 0, 0
            auc_0, auc_1, auc_2, auc_3, auc_4 = 0, 0, 0, 0, 0
            for i in range(crossval):
                X_train, y_train = generate_expert_data(n_samp=samps, noise_std=nois, plot=plot)
                X_test, y_test = generate_expert_data(n_samp=200, noise_std=nois)
                X_train = X_train[y_train.ravel() == 1]
                if len(X_train) == 0:
                    print(f"at iteration {i} no positive labels were created")
                    continue

                y_preds = fit_model(X_train, X_test, model)

                if model == 'gp':
                    # print("mean",y_preds[0], auc_score(y_test, y_preds[0]))
                    # print("var",y_preds[1], auc_score(y_test, y_preds[1]))
                    # print("pred",y_preds[2], auc_score(y_test, y_preds[2]))
                    # print("ratio",y_preds[3], auc_score(y_test, y_preds[3]))
                    pred_0 += (binary_accuracy(y_test, y_preds[0]))
                    auc_0 += auc_score(y_test, y_preds[0])
                    pred_1 += (binary_accuracy(y_test, y_preds[1]))
                    auc_1 += auc_score(y_test, y_preds[1])
                    pred_2 += (binary_accuracy(y_test, y_preds[2]))
                    auc_2 += auc_score(y_test, y_preds[2])
                    pred_3 += (binary_accuracy(y_test, y_preds[3]))
                    auc_3 += auc_score(y_test, y_preds[3])
                    pred_4 += (binary_accuracy(y_test, [1] * np.shape(y_preds)[1]))
                    auc_4 += auc_score(y_test, [1] * np.shape(y_preds)[1])
                else:
                    # print("svm",y_preds, auc_score(y_test, y_preds))
                    pred_0 += binary_accuracy(y_test, y_preds)
                    auc_0 += auc_score(y_test, y_preds)

            if model == 'gp':
                print("mean acc:", pred_0 / crossval, "AUC:", auc_0 / crossval)
                preds_0.append(pred_0 / crossval)
                aucs_0.append(auc_0 / crossval)

                print("var acc:", pred_1 / crossval, "AUC:", auc_1 / crossval)
                preds_1.append(pred_1 / crossval)
                aucs_1.append(auc_1 / crossval)

                print("pred acc:", pred_2 / crossval, "AUC:", auc_2 / crossval)
                preds_2.append(pred_2 / crossval)
                aucs_2.append(auc_2 / crossval)

                print("ratio acc:", pred_3 / crossval, "AUC:", auc_3 / crossval)
                preds_3.append(pred_3 / crossval)
                aucs_3.append(auc_3 / crossval)

                print("constant_1 acc:", pred_4 / crossval, "AUC:", auc_4 / crossval)
                preds_4.append(pred_4 / crossval)
                aucs_4.append(auc_4 / crossval)

            else:
                print("svm acc:", pred_0 / crossval, "AUC:", auc_0 / crossval)
                preds_0.append(pred_0 / crossval)
                aucs_0.append(auc_0 / crossval)

    if model == 'gp':
        acc_dict.update({"Mean": preds_0})
        auc_dict.update({"Mean": aucs_0})
        acc_dict.update({"Var": preds_1})
        auc_dict.update({"Var": aucs_1})
        acc_dict.update({"Pred": preds_2})
        auc_dict.update({"Pred": aucs_2})
        acc_dict.update({"Ratio": preds_3})
        auc_dict.update({"Ratio": aucs_3})
        acc_dict.update({"Constant": preds_4})
        auc_dict.update({"Constant": aucs_4})
    else:
        acc_dict.update({"SVM": preds_0})
        auc_dict.update({"SVM": aucs_0})
    return acc_dict, auc_dict


n_samples = [10, 50, 100, 500, 1000]  # , 2500]#, 10000]
n_samples = [5, 10, 50]
n_samples = [1000]

# n_samples = [100]
# noise_vals = [0.1, 1, 2, 5, 10, 20, 30, 100]
noise_vals = [0.001, 0.01, 1, 2, 5, 10]
# noise_vals = [1]
plot = False

crossval = 20
acc_dict, auc_dict = {}, {}
acc_dict, auc_dict = run_simulation(acc_dict, auc_dict, noise_vals, n_samples, crossval, plot, model='gp')
acc_dict, auc_dict = run_simulation(acc_dict, auc_dict, noise_vals, n_samples, crossval, plot, model='svm')
fig_url = "C:/Users/indy.dolmans/OneDrive - Nelen & Schuurmans/Pictures/simulations/"

if len(n_samples) == 1:  # if constant number of samples, we modify the noise parameter
    for key, value in acc_dict.items():
        print(key, value)
        plt.scatter(noise_vals, value, marker='x')
        plt.plot(noise_vals, value, label=key)
    # plt.xticks(list(range(len(noise_vals))), labels=[str(nv) for nv in noise_vals])
    plt.xlabel("Sample noise")
    figname = 'OC_Accuracy_Noise'
    plt.title(f'{n_samples=}, {crossval=}')
else:  # else, we modify the n_samples parameter
    for key, value in acc_dict.items():
        # plt.plot(list(range(len(n_samples))), value, label=key)
        plt.scatter(n_samples, value, marker='x')
        plt.plot(n_samples, value, label=key)

    # plt.xticks(list(range(len(n_samples))), labels=[str(samp) for samp in n_samples])
    plt.xlabel("Number of samples")
    figname = 'OC_Accuracy_nsamples'
    plt.title(f'{noise_vals=}, {crossval=}')
plt.ylabel('Prediction accuracy')
plt.legend()
plt.savefig(fig_url+figname)

plt.show()
if len(n_samples) == 1:  # if constant number of samples, we modify the noise parameter
    for key, value in auc_dict.items():
        print(key, value)
        plt.plot(list(range(len(noise_vals))), value, label=key)
    plt.xticks(list(range(len(noise_vals))), labels=[str(nv) for nv in noise_vals])
    plt.xlabel("Sample noise")
    figname = 'OC_AUC_Noise'
    plt.title(f'{n_samples=}, {crossval=}')
else:  # else, we modify the n_samples parameter
    for key, value in auc_dict.items():
        print(key, value)

        plt.plot(list(range(len(n_samples))), value, label=key)
    plt.xticks(list(range(len(n_samples))), labels=[str(samp) for samp in n_samples])
    plt.xlabel("Number of samples")
    figname = 'OC_AUC_nsamples'
    plt.title(f'{noise_vals=}, {crossval=}')
plt.ylabel('AUC')
plt.legend()
plt.savefig(fig_url+figname)
plt.show()
