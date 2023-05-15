from models import ml1_traintest, gp_model
from old_code import gpb2


def run_model(train_mod, test_mod, model, train_w, train_h, test_w, test_h, ref_std):
    if model == 'gpb':
        gpb2.run_model(train_mod, train_w, train_h, ref_std)
    elif model == 'gp':
        gp_model.run_model(train_mod, ref_std)
    else:
        ml1_traintest.run_model(train_mod, test_mod, model, train_w, train_h, test_w, test_h, ref_std)
