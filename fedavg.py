import copy
import torch

def fedavg(models):
    """
    Federated Averaging:
    Average the weights of the given local models.
    """
    avg_model = copy.deepcopy(models[0])
    avg_state_dict = avg_model.state_dict()

    # Initialize average with zeros
    for key in avg_state_dict.keys():
        avg_state_dict[key] = torch.zeros_like(avg_state_dict[key])

    # Sum model params
    for model in models:
        local_state_dict = model.state_dict()
        for key in avg_state_dict.keys():
            avg_state_dict[key] += local_state_dict[key]

    # Average params
    for key in avg_state_dict.keys():
        avg_state_dict[key] = avg_state_dict[key] / len(models)

    avg_model.load_state_dict(avg_state_dict)
    return avg_model

# Averaging is done layer wise here 