from audlib.sig.fbanks import MelFreq, ConstantQ
from audlib.quickstart import welcome
from audlib.sig.window import hamming
from audlib.sig.transform import stmfcc

import numpy as np
import scipy.signal as signal

sig, sr = welcome()


def test_mfcc():
    # TODO: need to add proper testing.
    nfft = 512
    nmel = 40
    melbank = MelFreq(sr, nfft, nmel)
    window_length = 0.032
    wind = hamming(int(window_length*sr))
    hop = .25
    mfcc = stmfcc(sig, wind, hop, nfft, melbank)
    return mfcc


def test_cqt():
    """Test constant Q transform."""
    nbins_per_octave = 32
    fmin = 100
    cqbank = ConstantQ(sr, fmin, bins_per_octave=nbins_per_octave)
    frate = 100
    cqt_sig = cqbank.cqt(sig, frate)

    return

def test_fbs():
    """Test filterbank synthesis."""
    window_length = 0.02
    window_size = int(window_length * sr)
    window = hamming(window_size, nchan=window_size, synth=True)
    synth = np.zeros(sig.shape, dtype=np.complex_)
    for kk in range(window_size):
        wk = 2 * np.pi * (kk / window_size)
        band = signal.lfilter(
            window * np.exp(1j*wk*np.arange(window_size)), 1, sig
        )
        synth[:] = synth[:] + band

    assert np.allclose(synth.real, sig)
    return


if __name__ == '__main__':
    test_fbs()
    test_mfcc()
    test_cqt()
