import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import geopandas as gpd
import configparser
import os

config = configparser.ConfigParser()
parent = os.path.dirname
config.read(os.path.join(parent(parent(__file__)), 'config.ini'))

data_url = config['DEFAULT']['data_url']


def labels_boxplot():
    expert_ref_source = 'expertscores_WSall'
    data = pd.read_csv(data_url + f'/{expert_ref_source}.csv', header=[0])

    df_mean = data.groupby(['Point'])['Value'].std().to_frame(name='Value_mean').reset_index()
    data = data.merge(df_mean, on=['Point'], how='left')

    unique_labels = data['Point'].unique()

    fig, ax = plt.subplots()
    boxplot_data = []
    for i, label in enumerate(unique_labels, start=1):
        label_data = data[data['Point'] == label]['Value_mean']
        boxplot_data.append(label_data)
        x_positions = [i] * len(label_data)
        ax.scatter(x_positions, label_data, color='black', s=5, label=label, alpha=0.7)

    ax.boxplot(boxplot_data, positions=range(1, len(unique_labels) + 1), labels=unique_labels)

    ax.set_xlabel('Location')
    ax.set_ylabel('Suitability score')
    ax.set_title('Expert-assessed suitability scores')
    plt.show()
    # ax.boxplot([data[data['Point'] == 'C']['Value'], data[data['Point'] == 'D']['Value']],
    #            positions=[1, 2], labels=['C', 'D'])
    # ax.scatter([1] * len(data[data['Point'] == 'C']['Value']),
    #            data[data['Point'] == 'C']['Value'], color='black', label='C', alpha=0.7)
    # ax.scatter([2] * len(data[data['Point'] == 'D']['Value']),
    #            data[data['Point'] == 'D']['Value'], color='black', label='D', alpha=0.7)
    # # plt.xticks(ticks=range(1, len(labels) + 1), labels=labels, rotation=90)
    # # plt.yticks(ticks=[0, 1, 2, 3, 4, 5], labels=ylabels)
    # # plt.subplots_adjust(bottom=0.4, left=0.25)
    # ax.set_xlabel('Point')
    # ax.set_ylabel('Value')
    # # plt.title('Expert-determined factor importance')
    # # plt.savefig(fig_name, bbox_inches='tight')
    # plt.show()


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

    thresholds = [6, 6, 1000, 1.5, 1000]  # 6m primary; 6m regional; 100mm subsid; 1.5m waterdepth; 2.5m soil capacity
    plt.subplots()
    for i, col in enumerate(data.columns[2:-1]):
        excess = data.loc[data[col] > thresholds[i], [col, 'Lng', 'Lat']]  # .to_list()
        print("excess",excess, thresholds[i], data[col], col)
        X1 = excess['Lng'].to_numpy()
        X2 = excess['Lat'].to_numpy()
        exc = excess[col]
        # plt.hist(data[col], bins=20)
        # plt.title(col)
        # plt.ylim([0,500])
        # plt.show()
        gdf = gpd.GeoDataFrame(exc, geometry=gpd.points_from_xy(X1,X2), crs="EPSG:3857")
        gdf.to_file(data_url + f'/extremevalsHR_{col}')


plot_extremes_map('noordhollandHiRes')

# cr = [0.7699097885245292, 0.5320110540028515, 0.5092075471442127, 0.15772281632105126, 1.3059199948375715, 0.1897392558838688, 0.31532103869288824, 0.16024435702130677, 0.2566570900472926, 0.19014449512496193, 1.2459304708172418, 0.1125542032089509]
# print(np.mean(cr), np.std(cr), np.var(cr))