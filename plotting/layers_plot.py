import numpy as np
from matplotlib import pyplot as plt, cm, colors

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
from textwrap import wrap

from data_util import data_loader

loader = data_loader.DataLoader('noordholland', ref_std='testdata')
X, nans, lnglat, size, col_names = loader.preprocess_input()

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
colmap = LinearSegmentedColormap.from_list('colmap2', [[1, 0, 0, 1], [145 / 255, 210 / 255, 80 / 255, 1]])
try:
    plt.register_cmap(cmap=colmap)
except:
    pass

print(f"{X.shape}")
width = int(np.sqrt(X.shape[0]))
data = X.reshape((width, width, -1))
print(f"{data.shape}")

# Column labels
labels = ['Flooding risk of primary dikes',
          'Flooding risk of regional dikes',
          'Ground subsidence',
          'Bottlenecks excessive rainwater',
          'Soil water storage capacity']
labels = ['\n'.join(wrap(x, 20)) for x in labels]

X = range(0, data.shape[0])
Y = range(0, data.shape[0])
xs, ys = np.meshgrid(X, Y)

zs1 = data[:, :, 0]
zs2 = data[:, :, 1]
zs3 = data[:, :, 2]
zs4 = data[:, :, 3]
zs5 = data[:, :, 4]
z = np.zeros(zs1.shape)

fig = plt.figure()
ax2 = fig.add_subplot(projection='3d')
ax2.plot_surface(np.flip(xs), ys, zs5 + 100, shade=False, cmap='YlOrBr', alpha=1.0)
ax2.plot_surface(np.flip(xs), ys, z, color='#78bced', shade=False, alpha=0.6)

ax2.plot_surface(np.flip(xs), ys, zs4 + 1500, shade=False, cmap='Blues', alpha=1.0)
ax2.plot_surface(np.flip(xs), ys, z + 1400, color='#78bced', shade=False, alpha=0.6)

ax2.plot_surface(np.flip(xs), ys, zs3 + 3000, shade=False, cmap='YlOrBr', alpha=1.0)
ax2.plot_surface(np.flip(xs), ys, z + 2900, color='#78bced', shade=False, alpha=0.6)

ax2.plot_surface(np.flip(xs), ys, zs2 + 4500, shade=False, cmap='Blues', alpha=1.0)
ax2.plot_surface(np.flip(xs), ys, z + 4400, color='#78bced', shade=False, alpha=0.6)

ax2.plot_surface(np.flip(xs), ys, zs1 + 6000, shade=False, cmap='Blues', alpha=1.0)
ax2.plot_surface(np.flip(xs), ys, z + 5900, color='#78bced', shade=False, alpha=0.6)

ax2.set_zticks([0, 1500, 3000, 4500, 6000])
ax2.set_zticklabels(labels[::-1])

plt.show()
