from models import baseline_model, GP_model, OCGPR


def run_model(train_mod, test_mod, model, train_size, test_size, ref_std):
    if model == 'ocgp':
        OCGPR.run_model(train_mod, test_mod, train_size, test_size)
    elif model == 'gp':
        GP_model.run_model(train_mod, test_mod, test_size)
    else:
        baseline_model.run_model(train_mod, test_mod, model, train_size, test_size, ref_std)
