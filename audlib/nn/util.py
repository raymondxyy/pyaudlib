"""Utility functions for neural networks."""
import numpy as np
import torch


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
            print("{}[{}]\n{}".format('-' * 30, name, param.data.numpy()))


def numparams(model):
    """Calculate the total number of learnable parameters."""
    return sum(p.numel() for p in model.parameters())


class UnpackedSequence(object):
    """Unpack a PackedSequence to original (unpadded) examples."""
    def __init__(self, ps):
        """Construct an unpacked sequence object."""
        self.packed_sequence = ps
        lencnt = [int(n) for n in ps.batch_sizes[:-1]-ps.batch_sizes[1:]] \
            + [int(ps.batch_sizes[-1])]
        self.seqlengths = []  # seqlengths[i] contains length of example i
        for num, ll in zip(lencnt[::-1], range(len(lencnt), 0, -1)):
            self.seqlengths.extend([ll] * num)
        assert len(self.seqlengths) == self.packed_sequence.batch_sizes[0]

    def __len__(self):
        """Return number of examples in this batch."""
        return len(self.seqlengths)

    def __getitem__(self, i):
        """Get original idx-th item in the batch."""
        idx = torch.LongTensor(self.seqlengths[i])
        idx[0] = i
        idx[1:] = self.packed_sequence.batch_sizes[:self.seqlengths[i]-1]
        ei = self.packed_sequence.data[idx.cumsum(0)]  # example i
        return ei
