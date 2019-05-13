"""Test temporal functions."""
import numpy as np
from audlib.sig.temporal import convdn


def test_convdn():
    """Test convolution and downsampling."""
    sig = np.random.rand(50)
    h = np.random.rand(20)
    for mm in range(1, 30):
        out1 = np.convolve(sig, h)
        out1 = out1[::mm]
        out2 = convdn(sig, h, mm)
        assert np.allclose(out1, out2)


if __name__ == "__main__":
    test_convdn()
