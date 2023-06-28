import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.model_selection import GridSearchCV

from data_util import load_data
from plotting import plot

import configparser

config = configparser.ConfigParser()
config.read('config.ini')

plot_data = int(config['PLOTTING']['plot_data'])
plot_pred = int(config['PLOTTING']['plot_pred'])


def set_params():
    kernel = 'scaled'  # scaled adaptive se
    v = 1.0  # scaled   0.8
    N = 3  # scaled     4
    svar = 0.0045  # 0.0045
    ls = 2  # se        2
    p = 30  # adaptive  30
    return kernel, v, N, svar, ls, p


class OCGP():
    # https://github.com/AntonioDeFalco/Adaptive-OCGP
    def __init__(self):
        self.K = []
        self.Ks = []
        self.Kss = []
        self.L = []
        self.alpha = []
        self.v = []
        self.var = []

    def GPR_OCC(self, noise=0.01):

        self.K = self.K + noise * np.eye(np.size(self.K, 0), np.size(self.K, 1))
        self.Kss = self.Kss + noise * np.ones((np.size(self.Kss, 0), np.size(self.Kss, 1)))
        self.L = np.linalg.cholesky(self.K)
        self.alpha = np.linalg.solve(np.transpose(self.L), (np.linalg.solve(self.L, np.ones((np.size(self.K, 0), 1)))))

    def getGPRscore(self, modes):

        if modes == 'mean':
            score = np.dot(np.transpose(self.Ks), self.alpha)

        elif modes == 'var':
            if np.size(self.v) == 0:
                self.v = np.linalg.solve(self.L, self.Ks)
            score = [a + b for a, b in zip(-self.Kss, sum(np.multiply(self.v, self.v)))]

        elif modes == 'pred':
            if np.size(self.v) == 0:
                self.v = np.linalg.solve(self.L, self.Ks)
            if np.size(self.var) == 0:
                self.var = [a - b for a, b in zip(self.Kss, sum(np.multiply(self.v, self.v)))]
            score = -0.5 * (np.divide(
                np.power((np.ones((np.size(self.var, 0), 1))) - (np.dot(np.transpose(self.Ks), self.alpha)), 2),
                self.var) + np.log(np.multiply(2 * np.pi, self.var)))

        elif modes == 'ratio':
            if np.size(self.v) == 0:
                self.v = np.linalg.solve(self.L, self.Ks)
            if np.size(self.var) == 0:
                self.var = [a - b for a, b in zip(self.Kss, sum(np.multiply(self.v, self.v)))]
            score = np.log(np.divide(np.dot(np.transpose(self.Ks), self.alpha), np.sqrt(self.var)))

        return score

    def seKernel(self, x, y, ls, svar=0.0045):
        self.K = svar * np.exp(-0.5 * cdist(x, x, 'sqeuclidean') / ls)
        self.Ks = svar * np.exp(-0.5 * cdist(x, y, 'sqeuclidean') / ls)
        self.Kss = svar * np.ones((np.size(y, 0), 1))

        self.GPR_OCC()

    def adaptiveKernel(self, x, y, ls, svar=0.0045):
        self.K = svar * np.exp(-0.5 * self.euclideanDistanceAdaptive(x, x, ls))
        self.K = (self.K + np.transpose(self.K)) / 2
        self.Ks = svar * np.exp(-0.5 * self.euclideanDistanceAdaptive(x, y, ls))
        self.Kss = svar * np.ones((np.size(y, 0), 1))
        self.GPR_OCC()

    def scaledKernel(self, x, y, v, meanDist_xn, meanDist_yn, svar=0.0045):

        self.K = svar * np.exp(-0.5 * self.euclideanDistanceScaled(x, x, v, meanDist_xn, meanDist_xn))
        self.Ks = svar * np.exp(-0.5 * self.euclideanDistanceScaled(x, y, v, meanDist_xn, meanDist_yn))
        self.Kss = svar * np.ones((np.size(y, 0), 1))
        self.GPR_OCC()

    def euclideanDistanceScaled(self, x, y, v, meanDist_xn, meanDist_yn):
        distmat = np.zeros((np.size(x, 0), np.size(y, 0)))
        for i in range(0, np.size(x, 0)):
            for j in range(0, np.size(y, 0)):
                dist = (x[i, :] - y[j, :])
                dist2 = np.dot(dist, dist)
                dist = np.sqrt(dist2)
                epsilon_ij = (meanDist_xn[i] + meanDist_yn[j] + dist) / 3
                buff = np.divide(dist2, (v * epsilon_ij))
                distmat[i, j] = buff
        return distmat

    def euclideanDistanceAdaptive(self, x, y, ls):
        distmat = np.zeros((np.size(x, 0), np.size(y, 0)))
        for i in range(0, np.size(x, 0)):
            for j in range(0, np.size(y, 0)):
                buff = (x[i, :] - y[j, :])
                buff = buff / ls[i]
                distmat[i, j] = np.dot(buff, buff)
        return distmat

    def euclideanDistance(self, x, y):
        distmat = np.zeros((np.size(x, 0), np.size(y, 0)))
        for i in range(0, np.size(x, 0)):
            for j in range(0, np.size(y, 0)):
                buff = (x[i, :] - y[j, :])
                distmat[i, j] = np.dot(buff, buff)
        return distmat

    def knn(self, data, k):
        neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(data)
        dist = neigh.kneighbors(data, k)
        return dist[0]

    def adaptiveHyper(self, x, p):
        dist = self.knn(x, p)
        ls = dist[:, p - 1]
        return ls

    def scaledHyper(self, x, y, N):
        dist_xn = self.knn(x, N)
        # dist_yn = self.knn(y,N)
        # dist_yn = distance.cdist(x, y)
        dist_yn = distance.cdist(y, x)  # (as MATLAB)
        dist_yn = np.sort(dist_yn, axis=1)
        dist_yn = dist_yn[:, 0:N]
        meanDist_xn = np.mean(dist_xn, 1)
        meanDist_yn = np.mean(dist_yn, 1)
        return meanDist_xn, meanDist_yn

    def preprocessing(self, x, y, mode, pca=False):

        if mode == "minmax":
            scaler = MinMaxScaler()
            all = np.vstack([x, y])
            scaler.fit(all)
            all = scaler.transform(all)
            x = all[0: np.size(x, 0), :]
            y = all[np.size(x, 0):np.size(all), :]
        elif mode == "zscore":
            scaler = StandardScaler()
            scaler.fit(x)
            x = scaler.transform(x)
            y = scaler.transform(y)

        if pca:
            pca = PCA(n_components=0.80)
            pca.fit(np.vstack([x, y]))
            x = pca.transform(x)
            y = pca.transform(y)

        return x, y


def run_model(train_mod, test_mod, train_w, train_h, test_w, test_h):
    kernel, v, N, svar, ls, p = set_params()
    X, Y, _, _ = load_data.load_xy(train_mod, model='hist_buildings')
    X_test, y_test, _, _ = load_data.load_xy(test_mod, model='hist_buildings')

    n_feats = 7
    X = X[:, -n_feats:]

    # width, height = 40, 40
    # X = X.reshape((width, height, n_feats))
    # Y = Y.reshape((width, height))
    # width, height = 20, 20
    # X = X[0:20, 20:40, :]
    # Y = Y[0:20, 20:40]
    # X = X.reshape((width*height, n_feats))
    # Y = Y.reshape((width*height,))

    train_vals = Y != 0
    X_train = X[train_vals]
    y_train = Y

    if plot_data:
        bg_train = load_data.load_bg(train_mod)
        bg_test = load_data.load_bg(test_mod)
        plot.plot_y(y_train, y_test, bg_train, bg_test, train_w, train_h, test_w, test_h)

    param_grid = {'v': [0.8, 0.4, 1.2],
                  'N': [4, 2, 8],
                  'svar': [0.0045, 0.001, 0.01],
                  'ls': [2, 1, 1/2],
                  'p': [30, 15, 50]}

    for key in param_grid:
        for i in range(3):
            v = param_grid['v'][0]
            N = param_grid['N'][0]
            svar = param_grid['svar'][0]
            ls = param_grid['ls'][0]
            p = param_grid['p'][0]


        ocgp = OCGP()

    if kernel == "se":
        ocgp.seKernel(X_train, X_test, ls, svar)
    if kernel == "adaptive":
        ls = ocgp.adaptiveHyper(X_train, p)
        ls = np.log(ls)
        # X_train, X_test = ocgp.preprocessing(X_train, X_test, "minmax",True)
        ocgp.adaptiveKernel(X_train, X_test, ls, svar)
    elif kernel == "scaled":
        # X_train, X_test = ocgp.preprocessing(X_train, X_test, "minmax",True)
        meanDist_xn, meanDist_yn = ocgp.scaledHyper(X_train, X_test, N)
        ocgp.scaledKernel(X_train, X_test, v, meanDist_xn, meanDist_yn, svar)

    modes = ['mean', 'var', 'pred', 'ratio']
    titles = [r'mean $\mu_*$', r'neg. variance $-\sigma^2_*$', r'log. predictive probability $p(y=1|X,y,x_*)$',
              r'log. moment ratio $\mu_*/\sigma_*$']

    X1 = X_test[:, 0]
    X2 = X_test[:, 1]
    for i in range(len(modes)):
        score = np.array(ocgp.getGPRscore(modes[i]))
        ax = plt.subplot(2, 2, i + 1)
        plot.set_colmap()
        img = plt.imshow(score.reshape((test_h, test_w)), vmin=np.nanmin(score),
                         vmax=np.nanmax(score), extent=[np.min(X1), np.max(X1), np.min(X2), np.max(X2)], aspect='auto')
        img.cmap.set_bad('tab:blue')

        print(modes[i], np.nanmin(score), np.nanmax(score))
        ax.set_title(titles[i])
    plt.tight_layout()
    plt.show()
