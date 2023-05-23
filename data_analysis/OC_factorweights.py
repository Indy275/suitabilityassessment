import datetime

import configparser
import os
from data_analysis import ahp_util
import numpy as np
import pandas as pd

config = configparser.ConfigParser()
parent = os.path.dirname
config.read(os.path.join(parent(parent(__file__)), 'config.ini'))
data_url = config['DEFAULT']['data_url']

choice_data = 'Bodem en Watersturend OC_May 17, 2023_11.51'
numeric_data = 'Bodem en Watersturend OC_May 18, 2023_11.53'
cluster = 'OC'

df = ahp_util.read_survey_data(choice_data, datetime.date(2023, 5, 15))
data = ahp_util.read_survey_data(numeric_data, datetime.date(2023, 5, 15))

data.drop(data[data['Q1'] == 'Student Civiele Techniek (ik ben stagiair)'].index, inplace=True)
print(data['Q2_1'])
for col in ['Q2_1', 'Q2_2', 'Q2_3', 'Q2_4', 'Q2_5']:
    data[col] = pd.to_numeric(data[col])
    data[col] = np.where(data[col] == 6, 1, data[col])
overstrPrimm = data['Q2_1'].mean(axis=0)
overstrRegim = data['Q2_2'].mean(axis=0)
bodemdalingm = data['Q2_3'].mean(axis=0)
wtroverlastm = data['Q2_4'].mean(axis=0)
bodembergenm = data['Q2_5'].mean(axis=0)

print("Weging van factoren:", overstrPrimm, overstrRegim, bodemdalingm, wtroverlastm, bodembergenm)

overstrPrimstd = data['Q2_1'].std(axis=0)
overstrRegistd = data['Q2_2'].std(axis=0)
bodemdalingstd = data['Q2_3'].std(axis=0)
wtroverlaststd = data['Q2_4'].std(axis=0)
bodembergenstd = data['Q2_5'].std(axis=0)

print("Std weging van factoren:", overstrPrimstd, overstrRegistd, bodemdalingstd, wtroverlaststd, bodembergenstd)
means = [ overstrPrimm, overstrRegim, bodemdalingm, wtroverlastm, bodembergenm]
stds = [overstrPrimstd, overstrRegistd, bodemdalingstd, wtroverlaststd, bodembergenstd]
factors = ['overstrPrim', 'overstrRegi', 'bodemdaling', 'wtroverlast', 'bodembergen']
with open(data_url + "/factorweights_{}.csv".format(cluster), 'w') as f:
    f.write("Factor,Mean,Std\n")
    for p, mean, std in zip(factors, means, stds):
        f.write("%s,%s,%s\n" % (p, mean, std))
