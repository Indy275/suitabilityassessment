import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from data_util import data_loader
import geopandas as gpd


data_url = 'C:\\Users\indy.dolmans\Documents\data'


def CRvsDuration():
    df1 = pd.read_csv(data_url + '/expertscores_WS.csv')
    df2 = pd.read_csv(data_url + '/expertscores2_WS.csv')

    plt.scatter(df1['Value'], df2['Value'])
    plt.show()
    plt.scatter(df1['Std'], df2['Std'])
    plt.show()

    # Compare duration with consistency ratio
    df1 = pd.read_csv(data_url + '/crduration.csv')
    df1['CR'] = pd.to_numeric(df1['CR'])
    df1['Duration'] = pd.to_numeric(df1['Duration'])

    plt.scatter(df1['Duration'], df1['CR'])
    plt.xlabel('Time (s)')
    plt.ylabel('Consistency Ratio')
    plt.show()


def plot_extremes_map(modifier):
    data = pd.read_csv(data_url + '/' + modifier + "/testdata_noscale.csv")
    bg = data_loader.load_bg(modifier)
    ratio = bg.shape[0] / bg.shape[1]

    thresh = [7, 6, 80, .5, 2]
    plt.subplots()
    for i, col in enumerate(data.columns[2:-1]):
        excess = data.loc[data[col] > thresh[i], [col, 'Lng', 'Lat']]  # .to_list()
        # print(excess, thresh[i], data[col], col)
        X1 = excess['Lng'].to_numpy()
        X2 = excess['Lat'].to_numpy()
        exc = excess[col]
        # plt.imshow(bg, origin='upper', aspect=ratio)
        # plt.scatter(X1, X2, c=exc, s=10, edgecolors='black')
        # plt.show()
        gdf = gpd.GeoDataFrame(exc, geometry=gpd.points_from_xy(X1,X2), crs="EPSG:4326")
        gdf.to_file(f'excessivalues_{col}')

plot_extremes_map('noordholland')