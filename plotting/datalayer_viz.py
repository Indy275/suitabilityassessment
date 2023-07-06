import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
from textwrap import wrap

from data_util import data_loader


X_train, _, X, train_col_names = data_loader.load_data('purmerend')

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

X = range(0, 40)
Y = range(0, 40)
xs, ys = np.meshgrid(X, Y)

zs1 = data[:,:,4]
zs2 = data[:,:,3]
zs3 = data[:,:,2]
zs4 = data[:,:,1]
zs5 = data[:,:,0]

fig = plt.figure()
ax2 = fig.add_subplot(projection='3d')
plot = ax2.plot_surface(np.flip(xs), ys, zs1,  cmap='viridis')
plot = ax2.plot_surface(np.flip(xs), ys, zs2 + 1500, cmap='viridis')
plot = ax2.plot_surface(np.flip(xs), ys, zs3 + 3000,  cmap='viridis')
plot = ax2.plot_surface(np.flip(xs), ys, zs4 + 4500, cmap='viridis')
plot = ax2.plot_surface(np.flip(xs), ys, zs5 + 6000, cmap='viridis')
ax2.set_zticks([0, 1500, 3000, 4500, 6000])
ax2.set_zticklabels(labels[::-1])

plt.show()