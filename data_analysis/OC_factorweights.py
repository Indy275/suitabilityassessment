import datetime

import configparser
import os
from data_analysis import ahp_util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

for col in ['Q2_5','Q2_4','Q2_3','Q2_2','Q2_1']:
    data[col] = pd.to_numeric(data[col])
    data[col] = np.where(data[col] == 1, 0, data[col])
    data[col] = np.where(data[col] == 6, 1, data[col])  # Qualtrics error: 'Een beetje belangrijk' is seen as 6
    # for i, expert in enumerate(data.itertuples()):
    #     if int(getattr(expert, 'Q2_1')) == 1:  # reversely filled in by expert: flooding risk not important at all?
    #         data.iloc[i, data.columns.get_loc(col)] = np.abs(int(getattr(expert, col))-6)

overstrPrimm = data['Q2_1'].mean(axis=0).round(decimals=2)
overstrRegim = data['Q2_2'].mean(axis=0).round(decimals=2)
bodemdalingm = data['Q2_3'].mean(axis=0).round(decimals=2)
wtroverlastm = data['Q2_4'].mean(axis=0).round(decimals=2)
bodembergenm = data['Q2_5'].mean(axis=0).round(decimals=2)

print("    Weging van factoren:", overstrPrimm, overstrRegim, bodemdalingm, wtroverlastm, bodembergenm)

overstrPrimstd = data['Q2_1'].std(axis=0).round(decimals=2)
overstrRegistd = data['Q2_2'].std(axis=0).round(decimals=2)
bodemdalingstd = data['Q2_3'].std(axis=0).round(decimals=2)
wtroverlaststd = data['Q2_4'].std(axis=0).round(decimals=2)
bodembergenstd = data['Q2_5'].std(axis=0).round(decimals=2)

print("Std weging van factoren:", overstrPrimstd, overstrRegistd, bodemdalingstd, wtroverlaststd, bodembergenstd)
means = [overstrPrimm, overstrRegim, bodemdalingm, wtroverlastm, bodembergenm]
stds = [overstrPrimstd, overstrRegistd, bodemdalingstd, wtroverlaststd, bodembergenstd]
factors = ['overstrPrim', 'overstrRegi', 'bodemdaling', 'wtroverlast', 'bodembergen']
with open(data_url + "/factorweights_{}.csv".format(cluster), 'w') as f:
    f.write("Factor,Mean,Std\n")
    for p, mean, std in zip(factors, means, stds):
        f.write("%s,%0.3f,%0.3f\n" % (p, mean, std))

plt.boxplot(data[['Q2_1','Q2_2','Q2_3','Q2_4','Q2_5']],
            labels=['Overstr Prim', 'Overstr Reg', 'Bodemdaling', 'Wateroverlast', 'Bodemberging'])
plt.show()