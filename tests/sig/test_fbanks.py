from audlib.sig.fbanks import MelFreq
from audlib.io.audio import audioread
from audlib.sig.window import hamming
from audlib.sig.transform import stmfcc

sig, sr = audioread('samples/welcome16k.wav')


def test_mfcc():
    # TODO: need to add proper testing.
    nfft = 512
    nmel = 40
    melbank = MelFreq(sr, nfft, nmel)
    window_length = 0.032
    wind = hamming(int(window_length*sr))
    hop = .25
    mfcc = stmfcc(sig, sr, wind, hop, nfft, melbank)
    return mfcc


if __name__ == '__main__':
    test_mfcc()
