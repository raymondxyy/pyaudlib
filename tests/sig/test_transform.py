import numpy as np

from audlib.sig.transform import stft, istft
from audlib.sig.window import hamming
from audlib.sig.util import nextpow2
from audlib.quickstart import welcome

WELCOME, SR = welcome()


def test_stft():
    """Test STFT and iSTFT."""
    sig = WELCOME
    for zp in (True, False):
        for wlen in (0.02, 0.025, 0.032, 0.5):
            for hop in (0.25, 0.5):
                wind = hamming(int(wlen*SR), hop=hop, synth=True)
                nfft = nextpow2(len(wind))
                spec = stft(sig, wind, hop, nfft, synth=True, zphase=zp)
                sigsynth = istft(spec, wind, hop, nfft, zphase=zp)
                assert np.allclose(sig, sigsynth[:len(sig)])


if __name__ == '__main__':
    test_stft()
