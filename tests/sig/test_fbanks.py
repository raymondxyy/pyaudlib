from audlib.sig.fbanks import MelFreq, ConstantQ
from audlib.quickstart import welcome
from audlib.sig.window import hamming
from audlib.sig.transform import stmfcc

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


if __name__ == '__main__':
    test_mfcc()
    test_cqt()
