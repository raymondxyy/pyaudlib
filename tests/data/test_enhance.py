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
    wsjspeech = RandSample("/home/xyy/data/wsj0/", sr=sr,
                           minlen=3., maxlen=8.,
                           filt=lambda p: p.endswith('.wv1'))
    assert len(wsjspeech) == 34287


def test_additive():
    """Test for additive noise."""
    wsjspeech = RandSample("/home/xyy/data/wsj0/", sr=sr,
                           minlen=3., maxlen=8.,
                           filt=lambda p: p.endswith('.wv1'))
    noizeus = RandSample("/home/xyy/Documents/MATLAB/loizou/Databases/noise16k/",
                         sr=sr, minlen=3.)
    wsj_noizues = Additive(wsjspeech, noizeus)  # noisy speech dataset
    assert len(wsj_noizues) == len(wsjspeech)


if __name__ == '__main__':
    test_randsamp()
    test_additive()
