"""Test enhance functions."""
from audlib.sig.window import hamming
from audlib.enhance import SSFEnhancer
from audlib.io.audio import audioread, audiowrite


def test_ssf():
    sig, sr = audioread(
        '/home/xyy/Documents/SSF_Package_11k/sb01_Reverb0P5sec.in.11k.wav')
    wlen = .05
    hop = .2
    nfft = 1024
    wind = hamming(int(wlen*sr), hop=hop, synth=True)
    enhancer = SSFEnhancer(sr, wind, hop, nfft)
    sigsynth = enhancer(sig)
    audiowrite(sigsynth, sr, 'test-ssf.wav')


if __name__ == "__main__":
    test_ssf()
