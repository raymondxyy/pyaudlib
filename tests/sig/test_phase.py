"""Test suite for audlib.sig.phase."""
import numpy as np
import soundfile as sf

from audlib.sig.transform import stft, istft
from audlib.sig.window import hamming
from audlib.sig.util import nextpow2
from audlib.sig.spectral import magphasor
from audlib.sig.phase import griffin_lim
from audlib.quickstart import welcome

WELCOME, SR = welcome()


def test_griffin_lim():
    """Test STFT and iSTFT."""
    sig = WELCOME
    wlen = .032
    hop = 128
    wsize = int(wlen*SR)

    # modified Hamming window
    a = .54
    b = -.46
    phi = np.pi / wsize
    wind = 2/(4*a**2+2*b**2)**.5*(a+b*np.cos(2*np.pi*np.arange(wsize)/wsize+phi))


    wind = hamming(wsize, hop=hop)
    nfft = nextpow2(len(wind))
    spec = stft(sig, wind, hop, nfft)
    magspec, _ = magphasor(spec)

    sig, err = griffin_lim(magspec, wind, hop, nfft, 100, init=0)
    assert all(e1 >= e2 for (e1, e2) in zip(err[:-1], err[1:]))


if __name__ == '__main__':
    test_griffin_lim()
