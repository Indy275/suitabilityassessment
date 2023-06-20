import numpy as np
# import matplotlib as mpl
# mpl.rcParams.update(mpl.rcParamsDefault)
import matplotlib.pyplot as plt
from data_util import load_data
from scipy.spatial.distance import cdist


def se_kernel(loghypers, x, y=None):
    x = x.transpose()
    ls = np.exp(2 * loghypers[0])
    svar = np.exp(2 * loghypers[1])
    K = svar * np.exp(-0.5 * cdist(x, x, 'sqeuclidean') / ls)
    Ks = svar * np.exp(-0.5 * cdist(x, y, 'sqeuclidean') / ls)
    Kss = svar * np.ones((y.shape[0], 1))
    return K, Ks, Kss


def GPR_OCC(K, Ks, Kss, mode, kernel_centering=None):
    # only directly makes sense for mode='var'
    if kernel_centering is not None and mode == 'var':
        K, Ks, Kss = kcenter(K, Ks, Kss)

    noise = 0.01
    K = K + noise * np.eye(K.shape[0])
    Kss = Kss + noise * np.ones(Kss.shape)
    L = np.linalg.cholesky(K).T
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, np.ones(K.shape[0])))

    if mode == 'mean':
        score = np.dot(Ks.T, alpha)

    elif mode == 'var':
        v = np.linalg.solve(L, Ks)
        score = -Kss + np.sum(v * v, axis=0).reshape((-1, 1))

    elif mode == 'pred':
        v = np.linalg.solve(L, Ks)
        var = Kss - np.sum(v * v, axis=0).reshape((-1, 1))
        score = -0.5 * (((np.ones(var.shape[0]) - np.dot(Ks.T, alpha)) ** 2).reshape((-1, 1)) / var + np.log(
            2 * np.pi * var))

    elif mode == 'ratio':
        v = np.linalg.solve(L, Ks)
        score = np.log(np.dot(Ks.T, alpha).reshape((-1, 1)) / np.sqrt(Kss - np.sum(v * v, axis=0).reshape((-1, 1))))

    return score


def kcenter(K, Ks=None, Kss=None):
    n = K.shape[0]
    if Ks is None:
        Ks = np.ones((n, 1))
    if Kss is None:
        Kss = np.ones((1, 1))
    M = np.eye(n) - np.ones((n, n)) / n
    K = np.dot(np.dot(M, K), M)
    Ks = np.dot(M, Ks)
    Kss = np.dot(np.dot(Kss, M), M)
    return K, Ks, Kss


ls = -4
o_var = -5.5
ref_std = 'hist_buildings'
train_mod = 'purmerend'
width, height = 40, 40
X, Y, _, _ = load_data.load_xy(train_mod, ref_std)
X = Y.reshape((height, width))
X2 = np.nonzero(X)[0]
X1 = np.nonzero(X)[1]

Xrange = np.arange(np.min(X1) - 0.2 * (np.max(X1) - np.min(X1)), np.max(X1) + 0.2 * (np.max(X1) - np.min(X1)) + 0.01,
                   0.2)
Yrange = np.arange(np.min(X2) - 0.2 * (np.max(X2) - np.min(X2)), np.max(X2) + 0.2 * (np.max(X2) - np.min(X2)) + 0.01,
                   0.2)
print(f"{ np.min(X1)=} {np.max(X1)=} { np.min(X2)=} {np.max(X2)=}")

test = np.zeros((len(Xrange) * len(Yrange), 2))
c = 0
for i in Xrange:
    for j in Yrange:
        test[c, :] = [i, j]
        c += 1

print(f"{test.shape=} {len(Xrange)=} {len(Yrange)=}")

K, Ks, Kss = se_kernel([ls, o_var], np.vstack((X1, X2)), test)
print(f"{K.shape=} {Ks.shape=} {Kss.shape=}")

modes = ['mean', 'var', 'pred', 'ratio']
titles = [r'mean $\mu_*$', r'neg. variance $-\sigma^2_*$', r'log. predictive probability $p(y=1|X,y,x_*)$',
          r'log. moment ratio $\mu_*/\sigma_*$']

for i in range(len(modes)):
    score = GPR_OCC(K, Ks, Kss, modes[i])
    print(f"{score.T=} {score.shape=}")

    ax = plt.subplot(2, 2, i + 1)
    ax.pcolormesh(Xrange, Yrange, score.reshape(len(Yrange), len(Xrange)), cmap='gray')
    ax.set_title(titles[i])
    ax.set_xlim([min(Xrange), max(Xrange)])
    ax.set_ylim([min(Yrange), max(Yrange)])

plt.tight_layout()
plt.show()
