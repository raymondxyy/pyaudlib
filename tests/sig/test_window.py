import numpy as np
from audlib.sig.window import hamming, tcola, fcola


def test_hamming():
    """Test window functions"""
    wlen = 257
    hopfrac = .25
    window = hamming(wlen, hop=hopfrac, synth=True)
    assert np.allclose(tcola(window, hopfrac), 1)
    window = hamming(wlen, nchan=wlen, synth=True)
    assert np.allclose(fcola(window, wlen), 1)


if __name__ == '__main__':
    test_hamming()
