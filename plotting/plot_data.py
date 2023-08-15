import matplotlib.pyplot as plt
import pandas as pd

import plotting.plot
from shapely.geometry import Point
from collections import Counter


def plot_y(loader, bg, ref_std):
    y = loader.y
    lnglat = loader.lnglat[y != 0]
    print(loader.bbox)
    # bbox = [float(i[0:-1]) for i in loader.bbox.split()]
    bbox = loader.bbox

    fig, ax = plt.subplots()
    try:
        width, height = bg.size
    except:
        width, height = bg.shape[0], bg.shape[1]
    ratio = height / width
    ax.imshow(bg, extent=[bbox[0], bbox[2], bbox[1], bbox[3]], origin='upper', aspect=ratio)

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
        df.sort_values(['y_mean'], inplace=True)
        # ax.scatter(df['X1'], df['X2'], c=df['y_mean'], cmap=cmap, s=df['Count'] * 25, edgecolors='black')

        if len(df) == 5:
            labs = [r'$\alpha$', r'$\beta$', r'$\gamma$', r'$\delta$', r'$\epsilon$']
            df['sym'] = labs
        else:
            markers = []
            for let in list('ABCDEFGHIJKLMNOPQRSTUVWXYZab'):
                # markers.append(f'"${let}$"')
                markers.append('${}$'.format(let))
            df['sym'] = markers
        for i, row in df.iterrows():
            ax.scatter(row['X1'], row['X2'], c='r', s=50, marker='o')
            ax.scatter(row['X1'], row['X2'], c='k', s=70, marker=row['sym'])

        # ax.scatter(df['X1'], df['X2'], c='r', marker=df['sym'], edgecolors='black')

    ax.set_title("{} labels: {} data points".format(loader.modifier, len(y[y != 0])))
    plt.show()
