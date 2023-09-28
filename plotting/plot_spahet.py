import pandas as pd
import matplotlib.pyplot as plt
import configparser
import numpy as np
import os
from joblib import load
from data_util import data_loader
import seaborn as sns

config = configparser.ConfigParser()
parent = os.path.dirname
config.read(os.path.join(parent(parent(__file__)), 'config.ini'))

data_url = config['DEFAULT']['data_url']

modifier1 = 'ws'
modifier2 = 'noordholland'
# Load the dataframe from CSV or any other data source
train_data = data_loader.DataLoader(modifier1, ref_std='expert_ref')
test_data = data_loader.DataLoader(modifier2, ref_std='testdata')

X_train, y_train, train_lnglat, train_col_names = train_data.preprocess_input()
X_test, test_nans, test_lnglat, test_size, test_col_names = test_data.preprocess_input()

df1_orig = train_data.load_orig_df()
df2_orig = test_data.load_orig_df()

def plot_correlation():
    dfcols = ['Primair', 'Regionaal', 'Bodemdaling', 'Inundatie', 'Bodemberging']
    # sns.pairplot(pd.DataFrame(X_test, columns=test_col_names))
    df = pd.DataFrame(X_test, columns=test_col_names)
    plt.matshow(df.corr())
    plt.xticks(range(df.shape[1]), dfcols, fontsize=8, rotation=10)
    plt.yticks(range(df.shape[1]), dfcols, fontsize=8, rotation=80)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16);plt.show()

# plt.hist(df1_orig[:, 0])
# plt.show()
# Create a bar graph for each column
for i, column in enumerate(train_col_names):
    plt.figure()

    plt.hist(df1_orig[:,i], label=f'{modifier1}', alpha=0.5, color='g')
    plt.hist(df2_orig[:,i], label=f'{modifier2}', alpha=0.5, color='b')

    # num_bin = 50
    # bin_lims = np.linspace(0, np.nanmax(np.array([df1_orig[:,i],df2_orig[:,i]]).flatten()), num_bin + 1)
    # bin_centers = 0.5 * (bin_lims[:-1] + bin_lims[1:])
    # bin_centers *= np.max(df1_orig[:,i])
    # bin_widths = bin_lims[1:] - bin_lims[:-1]
    # hist1, edges1 = np.histogram(df1_orig[:,i], bins=bin_lims)
    # hist2, edges2 = np.histogram(df2_orig[:,i], bins=bin_lims)

    # hist1b = hist1 / np.max(hist1)
    # hist2b = hist2 / np.max(hist2)
    # plt.bar(edges1[:-1], hist1b, width=np.diff(edges1), edgecolor='black', label=f'{modifier1}', align='edge', alpha=0.4, color='g')
    # plt.bar(edges2[:-1], hist2b, width=np.diff(edges2), edgecolor='black', label=f'{modifier2}',align='edge', alpha=0.4, color='b')
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.title(f'{column}')
    plt.legend()
    plt.ylim([0, 2000])
    plt.show()
