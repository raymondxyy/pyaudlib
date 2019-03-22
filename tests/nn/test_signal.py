"""Test nn.signal."""
import numpy as np
import torch
import scipy.signal as sig
from audlib.nn.signal import hilbert


def test_hilbert():
    """Test Hilbert transform."""
    nums = np.random.rand(32, 10)  # a batch
    # Test a single example
    signp = nums[0]
    sigtc = torch.from_numpy(signp)
    hilbnp = sig.hilbert(signp)
    hilbtc = hilbert(sigtc).numpy()
    assert np.allclose(hilbnp, hilbtc[:, 0]+1j*hilbtc[:, 1])
    assert np.allclose(signp, hilbtc[:, 0])
    # Test a batch
    signp = nums
    sigtc = torch.from_numpy(signp)
    hilbnp = sig.hilbert(signp)
    hilbtc = hilbert(sigtc).numpy()
    assert np.allclose(hilbnp, hilbtc[:, :, 0]+1j*hilbtc[:, :, 1])
    assert np.allclose(signp, hilbtc[:, :, 0])


if __name__ == "__main__":
    test_hilbert()
