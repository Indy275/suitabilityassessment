from models import baseline_model, GP_model, OCGPR, expert_weights


def run_model(train_mod, test_mod, model, ref_std):
    if model == 'ocgp':
        OCGPR.run_model(train_mod, test_mod)
    elif model == 'gp':
        GP_model.run_model(train_mod, test_mod)
    elif model == 'expert':
        expert_weights.run_model(test_mod)
    else:
        baseline_model.run_model(train_mod, test_mod, model, ref_std)
