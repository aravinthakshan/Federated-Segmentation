import copy
import torch

def fedsgd(global_model, local_gradients):
    """
    Federated SGD:
    Average the gradients from all clients and apply them to the global model.

    local_gradients: list of state_dict gradients from clients
    global_model: the model to update

    Returns the updated global_model.
    """
    avg_grads = {}

    # Initialize average grads to zeros
    for key, val in global_model.state_dict().items():
        avg_grads[key] = torch.zeros_like(val)

    # Sum all gradients
    for grads in local_gradients:
        for key in grads.keys():
            avg_grads[key] += grads[key]

    # Average gradients
    for key in avg_grads.keys():
        avg_grads[key] /= len(local_gradients)

    # Update global model weights: w = w - lr * avg_grad
    # Here we assume learning rate is applied during client-side training,
    # so we simply do a gradient step with the averaged gradients.
    updated_state_dict = global_model.state_dict()
    for key in updated_state_dict.keys():
        updated_state_dict[key] -= avg_grads[key]

    global_model.load_state_dict(updated_state_dict)
    return global_model
