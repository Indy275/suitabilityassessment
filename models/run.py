from models import ml1_traintest, gp_model, GPyOCC, gp_try3, OCC_GPR2
from old_code import gpb2


def run_model(train_mod, test_mod, model, train_w, train_h, test_w, test_h, ref_std):
    if model == 'gpb':
        gpb2.run_model(train_mod, train_w, train_h, ref_std)
    elif model == 'gpold':
        if ref_std == 'hist_buildings':
            GPyOCC.run_model(train_mod, test_mod, train_w, train_h, test_w, test_h)
        else:  # expert_ref
            gp_model.run_model(train_mod, ref_std)
    elif model == 'gpnew':
        if ref_std == 'hist_buildings':
            OCC_GPR2.run_model(train_mod, test_mod, train_w, train_h, test_w, test_h)
        else:  # expert_ref
            gp_try3.run_model(train_mod, test_mod, test_w, test_h)
    else:
        ml1_traintest.run_model(train_mod, test_mod, model, train_w, train_h, test_w, test_h, ref_std)
