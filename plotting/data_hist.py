import pandas as pd
import matplotlib.pyplot as plt
import configparser
import numpy as np
import os
from joblib import load
from data_util import data_loader

config = configparser.ConfigParser()
parent = os.path.dirname
config.read(os.path.join(parent(parent(__file__)), 'config.ini'))

data_url = config['DEFAULT']['data_url']

modifier = 'noordholland'
dataloader = data_loader.DataLoader(modifier, ref_std='hist_buildings')
df = pd.DataFrame(dataloader.load_orig_df(), columns=dataloader.col_names)
print(df.columns)

poor_houses = df[df['inundatiediepte T100']> 0.8][['Lat', 'Lng']]
print('inundatiediepte T100', poor_houses)
poor_houses = df[df['Overstromingsbeeld primaire keringen']> 5][['Lat', 'Lng']]
print('Overstromingsbeeld primaire keringen', poor_houses)

# Create a bar graph for each column
thresholds = [55,55,7, 6, 80, .5, 2]
extreme_vals = []
for i, column in enumerate(df.columns): #[2:]
    plt.figure()
    dft = df.loc[df[column] > thresholds[i], [column, 'Lng', 'Lat']]  # .to_list()
    print(column, f'has max value {np.nanmax(df[column])} and {len(dft)} cases exceeding threshold {thresholds[i]}')
    X1 = dft['Lng']
    X2 = dft['Lat']
    exc = dft[column]
    print(X1, X2, exc)
    plt.hist(df[column], bins=40)
    plt.title(f'{column}: maximum value: {np.nanmax(df[column])}.')#\n {len(dft)} cases exceed threshold: {thresholds[i]}')
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.ylim([0, 500])
    plt.show()


