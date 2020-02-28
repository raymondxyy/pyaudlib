import os
import pytest
from audlib.data.enhance import RandSample, Additive
from audlib.sig.window import hamming

# Pre-determined transform to be applied on signal
sr = 16000
window_length = 0.032
hopfrac = 0.25
wind = hamming(int(window_length*sr), hop=hopfrac)
nfft = 512


def test_randsamp():
    """Test for random sampling class."""
    #TODO
    pass


def test_additive():
    """Test for additive noise."""
    #TODO
    pass


if __name__ == '__main__':
    test_randsamp()
    test_additive()
