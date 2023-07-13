from data_util import create_data
from plotting.interactive_plot import DashApp
from util import geospatial
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

# todo: read bbox and size from metadata file
bbox = {'schermerbeemster': '4.78729, 52.59679, 4.88033, 52.53982',
        'purmer': '5.04805, 52.52374, 4.93269, 52.46846',
        'purmerend': '4.90, 52.4800, 4.96284, 52.51857',
        'volendam': '5.0168, 52.51664, 5.0853, 52.48006',
        'volendamLowRes': '5.0168, 52.51664, 5.0853, 52.48006',
        'oc': '4.8998, 52.4769, 5.0386, 52.5355',
        'ocall': '4.8998, 52.4769, 5.0386, 52.5355',
        'ws': '4.8281, 52.5892, 5.0457, 52.714',
        'wsall': '4.8281, 52.5892, 5.0457, 52.714',
        'tester': '4.8281, 52.5892, 5.0457, 52.714'}

#todo deal with noordholland not having bbox but required in metadata
if train_mod.startswith('noordholland') or test_mod.startswith('noordholland'):
    HHNKbbox = geospatial.get_hhnk_bbox()
    bbox['noordholland'] = HHNKbbox
    bbox['noordhollandHiRes'] = HHNKbbox


train_bbox = [float(i[0:-1]) for i in bbox[train_mod].split()]
test_bbox = [float(i[0:-1]) for i in bbox[test_mod].split()]

size = {'purmerend': 40, 'volendamLowRes': 40, 'volendam': 1000, 'ws': 1000,
        'noordholland': 1000, 'noordhollandHiRes': 2000, 'tester': 200}

train_size = size.get(train_mod, 400)
test_size = size.get(test_mod, 400)

if ml_model in ['gbr', 'gp']:
    ref_std = 'expert_ref'
elif ml_model in ['svm', 'ocgp']:
    ref_std = 'hist_buildings'
elif ml_model == 'expert':
    assert run_interactive_plot and not run_models, 'When model=expert, only interactive plot can be run'
else:
    assert ml_model in ['gbr', 'gp', 'svm', 'ocgp', 'expert'], f'ML model {ml_model} is invalid;' \
                                                     f' should be one of [gbr, gp, svm, ocgp, expert]'

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
