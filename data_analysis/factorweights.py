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

cluster = 'OC'
if cluster == 'WS':
    numeric_data = 'Bodem en Watersturend WS v2_June 9, 2023_09.44'
    start_date = datetime.date(2023, 5, 23)
    locationIDs = list('ABCDEFGHIJKLMNOPQRSTUVWXYZab')
else:# cluster == 'OC':
    numeric_data = 'Bodem en Watersturend OC_May 31, 2023_15.14'
    start_date = datetime.date(2023, 5, 15)
    locationIDs = list('ABCDE')

data = ahp_util.read_survey_data(numeric_data, datetime.date(2023, 5, 23))
data.dropna(subset=['Q1'], inplace=True)  # Drop all reports in which question was not answered

for col in ['Q2_5', 'Q2_4', 'Q2_3', 'Q2_2', 'Q2_1']:
    data[col] = pd.to_numeric(data[col])
    data[col] = np.where(data[col] == 1, 0, data[col])
    data[col] = np.where(data[col] == 6, 1, data[col])  # Qualtrics error: 'Een beetje belangrijk' received value 6

arit_means, geom_means, medians, stds = [], [], [], []
for col in ['Q2_1', 'Q2_2', 'Q2_3', 'Q2_4', 'Q2_5']:
    arit_means.append(data[col].mean(axis=0).round(decimals=3))
    geom_means.append(ahp_util.geo_mean(data[col]).round(decimals=3))
    medians.append(data[col].median(axis=0).round(decimals=3))
    stds.append(data[col].std(axis=0).round(decimals=3))

print("Arithmetic mean:   ", arit_means)
print("Geometric mean:    ", geom_means)
print("Median:            ", medians)
print("Standard deviation:", stds)

factors = ['overstrPrim', 'overstrRegi', 'bodemdaling', 'wtroverlast', 'bodembergen']
with open(data_url + "/factorweights_{}.csv".format(cluster), 'w') as f:
    f.write("Factor,Mean,Geomean,Median,Std\n")
    for p, amean, gmean, mdn, std in zip(factors, arit_means, geom_means, medians, stds):
        f.write("%s,%0.3f,%0.3f,%0.3f,%0.3f\n" % (p, amean, gmean, mdn, std))

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
ylabels = ['\n'.join(wrap(y, 10)) for y in ylabels]
fig_url = 'C://Users/indy.dolmans/OneDrive - Nelen & Schuurmans/Pictures/'
fig_name = fig_url + cluster + '_factorweights'

plt.boxplot(data[['Q2_1', 'Q2_2', 'Q2_3', 'Q2_4', 'Q2_5']])
plt.xticks(ticks=range(1, len(labels) + 1), labels=labels, rotation=90)
plt.yticks(ticks=[0, 1, 2, 3, 4, 5], labels=ylabels)
plt.subplots_adjust(bottom=0.4, left=0.25)
plt.xlabel('Soil and water factors')
plt.ylabel('Weighting')
plt.title('Expert-determined factor importance')
plt.savefig(fig_name, bbox_inches='tight')
plt.show()
