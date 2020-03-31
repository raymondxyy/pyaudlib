"""Torch-style transform functions."""

import numpy as np
import torch


class ToTensor(object):
    """Convert a numpy array to tensor. A direct copy from PyTorch."""

    def __init__(self, dtype):
        """`dtype` is typically set to `float32` for GPU."""
        self.dtype = dtype

    def __call__(self, featlst):
        """Convert a (possibly iterable of) np.ndarray to torch.tensor.

        Parameters
        ----------
        featlst: iterable of ndarrays

        Returns
        -------
        an iterable of tensors

        """
        def _to_tensor(feat):
            return torch.from_numpy(feat.astype(self.dtype))

        if isinstance(featlst, np.ndarray):
            return _to_tensor(featlst)
        elif isinstance(featlst, (tuple, list)):
            return tuple(map(_to_tensor, featlst))
        elif isinstance(featlst, dict):
            return {k: _to_tensor(v) for k, v in featlst.items()}
        else:
            raise ValueError('Incomprehensible type!')


class ToDevice(object):
    """Cast a torch.tensor to specific type before passing to device.

    A wrapper of tensor.to(), with the option for batch processing.
    """

    def __init__(self, device, dtype=None):
        """Instantiate a ToVariable callable."""
        super(ToDevice, self).__init__()
        self.device = device
        self.dtype = dtype
        if self.device.type == 'cuda':
            self.dtype = torch.float32

    def __call__(self, tensorlst):
        """Cast a tensor and send to target device."""
        def _to_device(tensor):
            return tensor.to(device=self.device, dtype=self.dtype)

        if isinstance(tensorlst, (tuple, list)):
            return tuple(map(_to_device, tensorlst))
        elif isinstance(tensorlst, dict):
            return {k: _to_device(v) for k, v in tensorlst.items()}
        else:
            raise ValueError('Incomprehensible type!')


class Compose(object):
    """Composes several transforms together. A direct copy from PyTorch."""

    def __init__(self, transforms):
        """Compose an iterable of transforms into one function."""
        self.transforms = transforms

    def __call__(self, sample):
        """Transform a sample."""
        for t in self.transforms:
            sample = t(sample)
        return sample
