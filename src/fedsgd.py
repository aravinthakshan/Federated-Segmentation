import copy
import torch

def fedsgd(global_model, local_gradients, lr):
    """
    Federated SGD:
    Average the gradients from all clients and apply them to the global model
    with learning rate scaling.

    Args:
        global_model: PyTorch model to be updated
        local_gradients: list of state_dict gradients from clients
        lr: learning rate scalar

    Returns:
        Updated global_model with averaged gradients applied.
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
    updated_state_dict = global_model.state_dict()
    for key in updated_state_dict.keys():
        updated_state_dict[key] -= lr * avg_grads[key]

    global_model.load_state_dict(updated_state_dict)
    return global_model
