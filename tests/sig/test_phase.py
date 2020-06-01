"""Test suite for audlib.sig.phase."""
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
    wlen = .025
    hop = .25
    wind = hamming(int(wlen*SR), hop=hop, synth=True)
    nfft = nextpow2(len(wind))
    spec = stft(sig, wind, hop, nfft, synth=True)
    magspec, _ = magphasor(spec)
    for zp in (True, False):
        phasor, err = griffin_lim(magspec, wind, hop, nfft, zphase=zp,
                                  reference=sig)
        print(err)
        sf.write(f"welcome-gl-zp_{zp}.wav", istft(magspec*phasor, wind, hop, nfft, zp), SR)


if __name__ == '__main__':
    test_griffin_lim()
