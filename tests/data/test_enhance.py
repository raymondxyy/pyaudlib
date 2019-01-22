from audlib.data.enhance import RandSample, Additive
from audlib.sig.transform import stlogm
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
                           mindur_per_file=3., sampdur_range=(3., 8.),
                           exts=('.wv1',))
    assert len(wsjspeech) == 34287

    numframes = 0
    for ii, samp in enumerate(wsjspeech):
        if not (ii+1 % 10):
            print(f"Processing [{ii+1}/{len(wsjspeech)}] files.")
        feat = stlogm(samp['data'], samp['sr'], wind, hopfrac, nfft)
        numframes += feat.shape[0]
        if (ii+1) == 100:
            break

    print(f"Total {numframes} frames of features computed.")


def test_additive():
    """Test for additive noise."""
    wsjspeech = RandSample("/home/xyy/data/wsj0/", sr=sr, mindur_per_file=3.,
                           sampdur_range=(3., 8.), exts=('.wv1',))
    noizeus = RandSample("/home/xyy/Documents/MATLAB/loizou/Databases/noise/",
                         sr=sr, mindur_per_file=3., sampdur_range=(3., None),
                         cache=True)
    wsj_noizues = Additive(wsjspeech, noizeus)  # noisy speech dataset
    assert len(wsj_noizues) == len(wsjspeech)

    numframes = 0
    for ii, samp in enumerate(wsj_noizues):
        if not (ii+1 % 10):
            print(f"Processing [{ii+1}/{len(wsj_noizues)}] files.")
        feat = stlogm(samp['chan1']['data'], samp['chan1']['sr'],
                      wind, hopfrac, nfft)
        numframes += feat.shape[0]
        if (ii+1) == 100:
            break

    print(f"Total {numframes} frames of features computed.")


if __name__ == '__main__':
    test_randsamp()
    test_additive()
