import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import geopandas as gpd
import configparser
import os

import plotting.plot

config = configparser.ConfigParser()
parent = os.path.dirname
config.read(os.path.join(parent(parent(__file__)), 'config.ini'))

data_url = config['DEFAULT']['data_url']


def labels_boxplot():
    cluster = 'WS'

    fig, ax = plt.subplots()
    expert_ref_source = f'expertscores_{cluster}all'
    data = pd.read_csv(data_url + f'/{expert_ref_source}.csv', header=[0])

    df_mean = data.groupby(['Point'])['Value'].std().to_frame(name='Value_mean').reset_index()
    data = data.merge(df_mean, on=['Point'], how='left')

    unique_labels = data['Point'].unique()
    if cluster == 'OC':
        labs = [r'$\alpha$', r'$\beta$', r'$\gamma$', r'$\delta$', r'$\epsilon$']
        ax.set_xlabel('Locations assessed by Research-cluster experts')
    else:
        labs = unique_labels
        ax.set_xlabel('Locations assessed by Water Systems-cluster experts')

    boxplot_data = []
    for i, label in enumerate(unique_labels, start=1):
        label_data = data[data['Point'] == label]['Value']#['Value_mean']
        boxplot_data.append(label_data)
        x_positions = [i] * len(label_data)
        ax.scatter(x_positions, label_data, color='black', s=5, label=labs[i-1], alpha=0.7)

    ax.boxplot(boxplot_data, positions=range(1, len(unique_labels) + 1), labels=labs)

    ax.set_ylabel('Suitability score')
    ax.set_title('Expert-assessed suitability scores per location')

    plt.show()


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


def plot_extremes_map():
    modifier = 'noordhollandHiRes'
    data = pd.read_csv(data_url + '/' + modifier + "/testdata_noscale.csv")

    thresholds = [6, 6, 1000, 1.5, 1000]  # 6m primary; 6m regional; 100mm subsid; 1.5m waterdepth; 2.5m soil capacity
    plt.subplots()
    for i, col in enumerate(data.columns[2:-1]):
        excess = data.loc[data[col] > thresholds[i], [col, 'Lng', 'Lat']]  # .to_list()
        print(len(excess),"excesses", thresholds[i])
        X1 = excess['Lat'].to_numpy()
        X2 = excess['Lng'].to_numpy()
        exc = excess[col]
        # plt.hist(data[col], bins=20)
        # plt.title(col)
        # plt.ylim([0,500])
        # plt.show()
        # print(min(X1), max(X1), min(X2), max(X2))
        gdf = gpd.GeoDataFrame(exc, geometry=gpd.points_from_xy(X2,X1), crs="EPSG:4326")
        # gdf.set_crs(epsg=4326, inplace=True)
        gdf.to_crs(epsg=28992, inplace=True)
        # print(gdf)
        # gdf.to_file(data_url + f'/NHextrvaluesHR_{col}')
        plt.scatter(X1, X2, c='#f81411', label='Inundation depth T100')
        plt.scatter(X1, X2, c='#1c7be7', label='Flooding risk regional dikes')
        plt.scatter(X1, X2, c='#becf50', label='Flooding risk primary dikes')
        plt.legend()
        plt.show()


def plot_sigdig():
    # values_1_to_2 = np.random.uniform(30, 50, 2)
    values_0_to_0_5 = np.random.normal(6, 0.5, 1000)  # Adjust the fraction as needed
    values_2_to_2_5 = np.random.normal(3, 1, 10000)  # Adjust the fraction as needed

    data = np.concatenate((values_0_to_0_5, values_2_to_2_5))
    data = plotting.plot.adjust_predictions(data,digitize=True, sigmoidal_tf=False)

    # n = 10
    # height = 1 / n
    # x_positions = range(n)
    # plt.bar(x_positions, [height] * n)
    # data = np.repeat(np.arange(0, n),10)

    plt.hist(data, edgecolor='k', density=True)
    plt.xlabel('Transformed suitability score')
    plt.ylabel('Percentage of values')
    # plt.xticks(x_positions, [f' {i + 1}' for i in range(n)])
    plt.show()

    # x = np.linspace(-10, 10, 100)
    # sigmoid = 1 / (1 + np.exp(-x))
    # plt.plot(x, sigmoid)
    # plt.xlabel('x')
    # plt.ylabel('Sigmoid(x)')
    # plt.title('Sigmoid Function')
    # plt.show()
    # print(data)

    # values_1_to_2 = np.random.uniform(30, 50, 2)
    # values_0_to_0_5 = np.random.normal(0, 1, 100 // 5)  # Adjust the fraction as needed
    # values_2_to_2_5 = np.random.normal(0, 1, 100 // 5)  # Adjust the fraction as needed
    #
    # data = np.concatenate((values_1_to_2, values_0_to_0_5, values_2_to_2_5))

    fig, ax = plt.subplots(1, 2)
    ax[0].hist(data, edgecolor='k')
    ax[0].set_title("Histogram before applying \n sigmoid transformation")
    sigmoid_f = lambda x: 1 / (1 + np.exp(-.5 * (x - .5)))
    ax[1].hist(sigmoid_f(data), edgecolor='k')
    ax[1].set_title("Histogram after applying \n sigmoid transformation")

    plt.show()


# plot_sigdig()
li  = [0.7699097885245292, 0.5320110540028515, 0.5092075471442127, 0.15772281632105126, 1.3059199948375715, 0.1897392558838688, 0.31532103869288824, 0.16024435702130677, 0.2566570900472926, 0.19014449512496193, 1.2459304708172418, 0.1125542032089509, 0.12285579688667925]
print(np.mean(li))

li = [0.9788424500270193, 0.18154497635817832, 0.20649344840226774, 1.054023806472518, 0.38735328955882664, 0.5267110814365034, 0.34026445488873247, 0.2881676028305806, 0.18973925588386661, 0.9434963311600033]
print(np.mean(li))
