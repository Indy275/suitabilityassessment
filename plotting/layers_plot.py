import numpy as np
from matplotlib import pyplot as plt, cm, colors

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
from textwrap import wrap

from data_util import data_loader

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256 ):
    """ mycolormap = truncate_colormap(
            cmap name or file or ndarray,
            minval=0.2, maxval=0.8 ): subset
            minval=1, maxval=0 )    : reverse
    by unutbu http://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
    """
    cmap = plt.get_cmap(cmap)
    name = "%s-trunc-%.2g-%.2g" % (cmap.name, minval, maxval)
    return colors.LinearSegmentedColormap.from_list(
        name, cmap( np.linspace( minval, maxval, n )))


X, col_names = data_loader.load_x('noordholland')

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

zs1 = data[:,:,2]
zs2 = data[:,:,3]
zs3 = data[:,:,4]
zs4 = data[:,:,5]
zs5 = data[:,:,6]

fig = plt.figure()
ax2 = fig.add_subplot(projection='3d')
plot = ax2.plot_surface(np.flip(xs), ys, zs1,  cmap='YlOrBr')
plot = ax2.plot_surface(np.flip(xs), ys, zs2 + 1500, cmap='Blues')
plot = ax2.plot_surface(np.flip(xs), ys, zs3 + 3000,  cmap='YlOrBr')
plot = ax2.plot_surface(np.flip(xs), ys, zs4 + 4500, cmap='Blues')
plot = ax2.plot_surface(np.flip(xs), ys, zs5 + 6000, cmap='Blues')
ax2.set_zticks([0, 1500, 3000, 4500, 6000])
ax2.set_zticklabels(labels[::-1])

plt.show()