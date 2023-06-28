from data_util import create_data
from plotting.interactive_plot import DashApp
from plotting import kde_plot
from models import run

import configparser

config = configparser.ConfigParser()
config.read('config.ini')

create_new_data = config.getboolean('RUN_SETTINGS', 'create_new_data')
run_interactive_plot = config.getboolean('RUN_SETTINGS', 'run_interactive_plot')
run_models = config.getboolean('RUN_SETTINGS', 'run_models')

train_mod = config['MODEL_SETTINGS']['train_mod']
test_mod = config['MODEL_SETTINGS']['test_mod']
ml_model = config['MODEL_SETTINGS']['ml_model']
data_url = config['DEFAULT']['data_url']

bbox = {'schermerbeemster': '4.78729, 52.59679, 4.88033, 52.53982',
        'purmer': '5.04805, 52.52374, 4.93269, 52.46846',
        'purmerend': '4.90, 52.4800, 4.96284, 52.51857',
        'noordholland': '4.51813, 52.45099, 5.2803, 52.96078',
        'NH_HiRes': '4.51813, 52.45099, 5.2803, 52.96078',
        'volendam': '5.0168, 52.51664, 5.0853, 52.48006',
        'oc': '4.8998, 52.4769, 5.0386, 52.5355',
        'ws': '4.8281, 52.5892, 5.0457, 52.714'}

train_bbox = [float(i[0:-1]) for i in bbox[train_mod].split()]
test_bbox = [float(i[0:-1]) for i in bbox[test_mod].split()]

size = {'purmerend': 40, 'noordholland': 1000, 'NH_HiRes': 2000, 'volendam': 1000, 'ws': 1000}

train_size = size.get(train_mod, 400)
test_size = size.get(test_mod, 400)

if ml_model in ['gbr', 'gp']:
    ref_std = 'expert_ref'
elif ml_model in ['svm', 'ocgp']:
    ref_std = 'hist_buildings'
else:
    assert ml_model in ['gbr', 'gp', 'svm', 'ocgp'], f'ML model {ml_model} is invalid;' \
                                                     f' should be one of [gbr, gp, svm, ocgp]'

if create_new_data:
    print("Creating data frame for {}".format(train_mod))
    create_data.create_df(train_mod, train_bbox, train_size, ref_std)
    print("Creating data frame for {}".format(test_mod))
    create_data.create_df(test_mod, test_bbox, test_size, ref_std)

if run_models:
    print("Running {}-model with {} for {} and {}".format(ml_model, ref_std, train_mod, test_mod))
    run.run_model(train_mod, test_mod, ml_model, train_size, test_size, ref_std)

if run_interactive_plot:
    print("Creating interactive plot")
    plot = DashApp(modifier=test_mod, size=test_size, w_model=ml_model)
    plot.run()
