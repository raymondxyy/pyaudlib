"""Utility Functions and Neural Networks."""
import numpy as np


def detach(states):
    """Truncate backpropagation (usually used in RNN)."""
    return [state.detach() for state in states]


def hasnan(m):
    """Check if torch.tensor m have NaNs in it."""
    return np.any(np.isnan(m.cpu().data.numpy()))


def printnn(model):
    """Print out neural network."""
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("{}[{}]\n{}".format('-'*30, name, param.data.numpy()))
