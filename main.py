from data_util import create_data, data_loader
from plotting.interactive_plot import DashApp
from models import run
from data_analysis import process_expert_data

import configparser

config = configparser.ConfigParser()
config.read('config.ini')

process_expertdata = config.getboolean('RUN_SETTINGS', 'process_expert')
create_new_data = config.getboolean('RUN_SETTINGS', 'create_new_data')
run_interactive_plot = config.getboolean('RUN_SETTINGS', 'run_interactive_plot')
run_models = config.getboolean('RUN_SETTINGS', 'run_models')

train_mod = config['MODEL_SETTINGS']['train_mod']
test_mod = config['MODEL_SETTINGS']['test_mod']
ml_model = config['MODEL_SETTINGS']['ml_model']
data_url = config['DEFAULT']['data_url']
cluster = config['EXPERT_SETTINGS']['cluster']
expert = config['EXPERT_SETTINGS']['expert']

if ml_model in ['gbr', 'gp']:
    ref_std = 'expert_ref'
elif ml_model in ['svm', 'ocgp']:
    ref_std = 'hist_buildings'
elif ml_model == 'expert':
    assert run_interactive_plot and not run_models, 'When model=expert, only interactive plot can be run'
else:
    assert ml_model in ['gbr', 'gp', 'svm', 'ocgp', 'expert'], f'ML model {ml_model} is invalid;' \
                                                     f' should be one of [gbr, gp, svm, ocgp, expert]'

load_train = data_loader.DataLoader(train_mod, ref_std)
load_test = data_loader.DataLoader(test_mod, ref_std='test_data')
train_bbox = [float(i[0:-1]) for i in load_train.bbox.split()]
test_bbox = [float(i[0:-1]) for i in load_test.bbox.split()]
train_size = load_train.size
test_size = load_test.size

if process_expertdata:
    process_expert_data.run_model(cluster, expert_processing=expert)

if create_new_data:
    print("     Creating data frame for {}".format(train_mod))
    create_data.create_df(train_mod, train_bbox, train_size, ref_std)
    print("     Creating data frame for {}".format(test_mod))
    create_data.create_df(test_mod, test_bbox, test_size, ref_std, test=True)

if run_models:
    print("     Running {}-model with {} for {} and {}".format(ml_model, ref_std, train_mod, test_mod))
    run.run_model(train_mod, test_mod, ml_model, train_size, test_size, ref_std)

if run_interactive_plot:
    print("     Creating interactive plot")
    plot = DashApp(modifier=test_mod, size=test_size, w_model=ml_model)
    plot.run()
