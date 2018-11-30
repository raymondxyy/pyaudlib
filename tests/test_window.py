import numpy as np 

from audlib.sig.window import hamming, cola
from audlib.io.audio import audioread

sig, fs = audioread('samples/welcome16k.wav')
window_length = 0.032
hop_fraction = 0.25
wind = hamming(int(window_length*fs), hop_fraction)
nfft = 512

def test_window():
    wlen = 256
    # np builtin hamming window has endpoint problem for OLA
    wind = hamming(wlen, hop=None)
    amp = cola(wind, hop_fraction)
    assert (amp is not None)
    wind = hamming(wlen, hop=hop_fraction)
    amp = cola(wind, hop_fraction)
    assert (amp is not None) and np.allclose(amp, 1.)