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

# Load the dataframe from CSV or any other data source
df = pd.read_csv(data_url + '/' + 'noordholland' + '/' + 'expert_ref' + ".csv")
ss = load(data_url + '/' + 'noordholland' + '/' + "scaler.joblib")
df2 = df.to_numpy()
df_orig = ss.inverse_transform(df2[:, 2:-1])
# Create a bar graph for each column
for i, column in enumerate(df.columns[2:-1]):
    plt.figure()
    # print(np.nanmax(df[column]), np.nanargmax(df[column]))
    # print(df_orig[np.nanargmax(df[column]),:])
    print("Column minmax:",np.nanmin(df_orig[:, i]),np.nanmax(df_orig[:, i]))

    plt.hist(df_orig[:, i], bins=20)
    plt.title(column)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.show()
