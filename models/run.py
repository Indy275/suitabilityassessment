from models import baseline_model, GP_model, OCGPR
from old_code import GPyOCC, gp_model


def run_model(train_mod, test_mod, model, train_size, test_size, ref_std):
    if model == 'gpold':
        if ref_std == 'hist_buildings':
            GPyOCC.run_model(train_mod, test_mod, train_size, test_size)
        else:  # expert_ref
            gp_model.run_model(train_mod, ref_std)
    elif model == 'gpnew':
        if ref_std == 'hist_buildings':
            OCGPR.run_model(train_mod, test_mod, train_size, test_size)
        else:  # expert_ref
            GP_model.run_model(train_mod, test_mod, test_size)
    else:
        baseline_model.run_model(train_mod, test_mod, model, train_size, test_size, ref_std)
