import pandas as pd
import matplotlib.pyplot as plt
import configparser
import numpy as np
import os
from joblib import load
config = configparser.ConfigParser()
parent = os.path.dirname
config.read(os.path.join(parent(parent(__file__)), 'config.ini'))

data_url = config['DEFAULT']['data_url']

modifier1 = 'ws'
modifier2 = 'noordholland'
# Load the dataframe from CSV or any other data source
df1 = pd.read_csv(data_url + '/' + modifier1 + "/testdata.csv")
ss1 = load(data_url + '/' + modifier1 + "/testdata_scaler.joblib")

df2 = pd.read_csv(data_url + '/' + modifier2 + "/testdata.csv")
ss2 = load(data_url + '/' + modifier2 + "/testdata_scaler.joblib")

df1_orig = ss1.inverse_transform(df1.iloc[:, 2:-1])
df2_orig = ss2.inverse_transform(df2.iloc[:, 2:-1])

# plt.hist(df1_orig[:, 0])
# plt.show()
# Create a bar graph for each column
for i, column in enumerate(df1.columns[2:-1]):
    plt.figure()

    # plt.hist(df1_orig[:,i], label=f'{modifier1}', alpha=0.5, color='g')
    # plt.hist(df2_orig[:,i], label=f'{modifier2}', alpha=0.5, color='b')

    num_bin = 50
    bin_lims = np.linspace(0, np.nanmax([df1_orig[:,i],df2_orig[:,i]]), num_bin + 1)
    # bin_centers = 0.5 * (bin_lims[:-1] + bin_lims[1:])
    # bin_centers *= np.max(df1_orig[:,i])
    # bin_widths = bin_lims[1:] - bin_lims[:-1]
    hist1, edges1 = np.histogram(df1_orig[:,i], bins=bin_lims)
    hist2, edges2 = np.histogram(df2_orig[:,i], bins=bin_lims)

    hist1b = hist1 / np.max(hist1)
    hist2b = hist2 / np.max(hist2)
    plt.bar(edges1[:-1], hist1b, width=np.diff(edges1), edgecolor='black', label=f'{modifier1}', align='edge', alpha=0.4, color='g')
    plt.bar(edges2[:-1], hist2b, width=np.diff(edges2), edgecolor='black', label=f'{modifier2}',align='edge', alpha=0.4, color='b')
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.title(f'{column}')
    plt.legend()
    # plt.ylim([0, 2000])
    plt.show()
