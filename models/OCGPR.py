import configparser
import numpy as np

from data_util import data_loader
from plotting import plot

from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

config = configparser.ConfigParser()
config.read('config.ini')

plot_data = int(config['PLOTTING']['plot_data'])
plot_pred = int(config['PLOTTING']['plot_pred'])


def set_hypers():
    kernel = config['MODEL_PARAMS_OCGP']['kernel']
    v = float(config['MODEL_PARAMS_OCGP']['v'])
    N = int(config['MODEL_PARAMS_OCGP']['N'])
    svar = float(config['MODEL_PARAMS_OCGP']['svar'])
    ls = float(config['MODEL_PARAMS_OCGP']['ls'])
    p = int(config['MODEL_PARAMS_OCGP']['p'])
    return kernel, v, N, svar, ls, p


class OCGP():
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


def run_model(train_mod, test_mod, train_size, test_size):
    X_train, y_train, _, _ = data_loader.load_data(train_mod, ref_std='hist_buildings')
    X_test, y_test, _, _ = data_loader.load_data(test_mod, ref_std='hist_buildings')

    n_feats = 7
    X = X_train[:, -n_feats:]

    # width, height = 40, 40
    # X = X.reshape((width, height, n_feats))
    # Y = Y.reshape((width, height))
    # width, height = 20, 20
    # X = X[0:20, 20:40, :]
    # Y = Y[0:20, 20:40]
    # X = X.reshape((width*height, n_feats))
    # Y = Y.reshape((width*height,))

    train_vals = y_train != 0
    X_train = X[train_vals]
    y_train = y_train

    if plot_data:
        bg_train = data_loader.load_bg(train_mod)
        bg_test = data_loader.load_bg(test_mod)
        plot.plot_y(y_train, y_test, bg_train, bg_test, train_size, test_size)

    kernel, v, N, svar, ls, p = set_hypers()
    ocgp = OCGP()

    if kernel == "se":
        title = f'SE kernel with ls={ls}, svar={svar}'
        ocgp.seKernel(X_train, X_test, ls, svar)
    if kernel == "adaptive":
        ls = ocgp.adaptiveHyper(X_train, p)
        ls = np.log(ls)
        title = f'Adaptive hyper with svar={svar}, p={p}, and learned ls={ls}'
        # X_train, X_test = ocgp.preprocessing(X_train, X_test, "minmax",True)
        ocgp.adaptiveKernel(X_train, X_test, ls, svar)
    elif kernel == "scaled":
        # X_train, X_test = ocgp.preprocessing(X_train, X_test, "minmax",True)
        meanDist_xn, meanDist_yn = ocgp.scaledHyper(X_train, X_test, N)
        title = f'Scaled hyper with svar={svar}, v={v}, N={N}'
        ocgp.scaledKernel(X_train, X_test, v, meanDist_xn, meanDist_yn, svar)

    modes = ['mean', 'var', 'pred', 'ratio']
    titles = [r'mean $\mu_*$', r'neg. variance $-\sigma^2_*$', r'log. predictive probability $p(y=1|X,y,x_*)$',
              r'log. moment ratio $\mu_*/\sigma_*$']

    X1 = X_test[:, 1]
    X2 = X_test[:, 0]
    cmap = plot.set_colmap()
    cmap.set_bad('tab:blue')
    bg = data_loader.load_bg(test_mod)
    for i in range(len(modes)):
        score = np.array(ocgp.getGPRscore(modes[i]))
        ax = plt.subplot(2, 2, i + 1)
        ax.imshow(bg, extent=[np.min(X1),np.max(X1),np.min(X2),np.max(X2)], origin='upper')
        plt.imshow(score.reshape((test_size, test_size)), vmin=np.nanmin(score), cmap=cmap, alpha=0.65,
                        vmax=np.nanmax(score), extent=[np.min(X1),np.max(X1),np.min(X2),np.max(X2)], aspect='auto')
        ax.set_title(titles[i])
    plt.suptitle(title)
    plt.tight_layout()
    fig_url = 'C://Users/indy.dolmans/OneDrive - Nelen & Schuurmans/Pictures/maps/'
    name = fig_url + test_mod + '_' + train_mod + '_ocgp_' + kernel
    plt.savefig(name, bbox_inches='tight')
    plt.show()
