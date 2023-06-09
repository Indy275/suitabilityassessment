import datetime

import configparser
import os
from data_analysis import ahp_util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from textwrap import wrap

config = configparser.ConfigParser()
parent = os.path.dirname
config.read(os.path.join(parent(parent(__file__)), 'config.ini'))
data_url = config['DEFAULT']['data_url']

# choice_data = 'Bodem en Watersturend OC_May 17, 2023_11.51'
numeric_data = 'Bodem en Watersturend OC_May 24, 2023_14.03'
cluster = 'OC'

# df = ahp_util.read_survey_data(choice_data, datetime.date(2023, 5, 15))
data = ahp_util.read_survey_data(numeric_data, datetime.date(2023, 5, 15))

# Drop student data
data.drop(data[data['Q1'] == 'Student Civiele Techniek (ik ben stagiair)'].index, inplace=True)
data.dropna(subset=['Q1'], inplace=True)  # Drop all reports in which question was not answered

# data = data[data['Q2_1'] != '1']  # Exclude the 'likely reversed answer'
for col in ['Q2_5', 'Q2_4', 'Q2_3', 'Q2_2', 'Q2_1']:
    data[col] = pd.to_numeric(data[col])
    data[col] = np.where(data[col] == 1, 0, data[col])
    data[col] = np.where(data[col] == 6, 1, data[col])  # Qualtrics error: 'Een beetje belangrijk' is seen as value 6
    # for i, expert in enumerate(data.itertuples()):  # Manipulate results of 'likely reversed answer'
    #   if int(getattr(expert, 'Q2_1')) == 1:
    #       data.iloc[i, data.columns.get_loc(col)] = np.abs(int(getattr(expert, col))-6)

arit_means, geom_means, stds = [], [], []
for col in ['Q2_1', 'Q2_2', 'Q2_3', 'Q2_4', 'Q2_5']:
    arit_mean = data[col].mean(axis=0).round(decimals=3)
    geom_mean = ahp_util.geo_mean(data[col]).round(decimals=3)
    std = data[col].std(axis=0).round(decimals=3)
    arit_means.append(arit_mean)
    geom_means.append(geom_mean)
    stds.append(std)

print("Arithmetic mean:   ", arit_means)
print("Geometric mean:    ", geom_means)
print("Standard deviation:", stds)

# factors = ['overstrPrim', 'overstrRegi', 'bodemdaling', 'wtroverlast', 'bodembergen']
# with open(data_url + "/factorweights_{}.csv".format(cluster), 'w') as f:
#     f.write("Factor,Mean,Std\n")
#     for p, amean, gmean, std in zip(factors, arit_means, geom_means, stds):
#         f.write("%s,%0.3f,%0.3f,%0.3f\n" % (p, amean, gmean, std))

labels = ['Flooding risk of primary embankments',
          'Flooding risk of regional embankments',
          'Ground subsidence',
          'Bottlenecks excessive rainwater',
          'Soil water storage capacity']
labels = ['\n'.join(wrap(x, 20)) for x in labels]
ylabels = ['Not important',
           'Somewhat important',
           'Moderately important',
           'Quite important',
           'Extremely important',
           'Absolutely important']
ylabels = ['\n'.join(wrap(x, 10)) for x in ylabels]

plt.boxplot(data[['Q2_1', 'Q2_2', 'Q2_3', 'Q2_4', 'Q2_5']])
plt.xticks(ticks=range(1, len(labels) + 1), labels=labels, rotation=90)
plt.yticks(ticks=[0, 1, 2, 3, 4, 5], labels=ylabels)
plt.subplots_adjust(bottom=0.4, left=0.25)
plt.xlabel('Soil and water factors')
plt.ylabel('Weighting')
plt.title('Expert-determined factor importance')
plt.show()
