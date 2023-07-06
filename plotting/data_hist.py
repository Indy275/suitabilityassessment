import pandas as pd
import matplotlib.pyplot as plt
import configparser
import numpy as np
import os
from joblib import load
from scipy.stats import zscore

config = configparser.ConfigParser()
parent = os.path.dirname
config.read(os.path.join(parent(parent(__file__)), 'config.ini'))

data_url = config['DEFAULT']['data_url']

modifier = 'noordholland'
# Load the dataframe from CSV or any other data source
df = pd.read_csv(data_url + '/' + modifier + "/testdata_noscale.csv")
ss = load(data_url + '/' + modifier + "/testdata_scaler.joblib")

# Create a bar graph for each column
thresholds = [7, 6, 80, .5, 2]
extreme_vals = []
for i, column in enumerate(df.columns[2:-1]):
    plt.figure()
    dft = df.loc[df[column] > thresholds[i], [column, 'Lng', 'Lat']]  # .to_list()
    print(column, f'has max value {np.nanmax(df[column])} and {len(dft)} cases exceeding threshold {thresholds[i]}')
    X1 = dft['Lng']
    X2 = dft['Lat']
    exc = dft[column]
    print(X1, X2, exc)
    plt.hist(df[column], bins=40)
    plt.title(f'{column}: max value: {round(np.nanmax(df[column]))}.\n {len(dft)} cases exceed threshold: {thresholds[i]}')
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.ylim([0, 500])
    plt.show()
