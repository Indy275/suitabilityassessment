import matplotlib.pyplot as plt
import pandas as pd

import plotting.plot
from shapely.geometry import Point
from collections import Counter


def plot_y(loader, bg, ref_std):
    y = loader.y
    lnglat = loader.lnglat[y != 0]
    bbox = [float(i[0:-1]) for i in loader.bbox.split()]

    fig, ax = plt.subplots()
    ax.imshow(bg, extent=[bbox[2], bbox[0], bbox[3], bbox[1]], origin='upper', alpha=0.8)

    if ref_std == 'hist_buildings':
        ax.scatter(lnglat[:, 0], lnglat[:, 1], s=1, c='r')
    elif ref_std == 'expert_ref':
        cmap = plotting.plot.set_colmap()
        points = [Point(x1, x2) for x1, x2 in loader.lnglat[:, :2]]
        point_counts = Counter(points)
        df = pd.DataFrame(
            [{'X1': p.x, 'X2': p.y, 'y': y[i], 'Count': point_counts[Point(p.x, p.y)]} for i, p in enumerate(points)])
        df_mean = df.groupby(['X1', 'X2'])['y'].mean().to_frame(name='y_mean').reset_index()
        df = df.merge(df_mean, on=['X1', 'X2'], how='left')
        ax.scatter(df['X1'], df['X2'], c=df['y_mean'], cmap=cmap, s=df['Count'] * 25, edgecolors='black')

    ax.set_title("{} labels: {} data points".format(loader.modifier, len(y[y != 0])))

    plt.show()
