from audlib.data.enhance import RandSample, Additive


def test_randsamp():
    """Test for random sampling class."""
    wsjspeech = RandSample("/home/xyy/data/wsj0/", sr=16000,
                           mindur_per_file=3., sampdur_range=(3., 8.),
                           exts=('.wv1',))
    print(len(wsjspeech))


def test_additive():
    """Test for additive noise."""
    SR = 16000  # fix sampling rate
    wsjspeech = RandSample("/home/xyy/data/wsj0/", sr=SR, mindur_per_file=3.,
                           sampdur_range=(3., 8.), exts=('.wv1',))
    noizeus = RandSample("/home/xyy/Documents/MATLAB/loizou/Databases/noise/",
                         sr=SR, mindur_per_file=3., sampdur_range=(3., None))
    wsj_noizues = Additive(wsjspeech, noizeus)  # noisy speech dataset
    print(len(wsj_noizues))


if __name__ == '__main__':
    test_randsamp()
    test_additive()
