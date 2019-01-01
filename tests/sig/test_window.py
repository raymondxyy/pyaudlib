from audlib.sig.window import hamming, tcola


def test_hamming():
    """Test window functions"""
    wlen = 256
    hopfrac = .25
    wind = hamming(wlen, hop=hopfrac, synth=True)
    amp = tcola(wind, hopfrac)
    assert amp == 1


if __name__ == '__main__':
    test_hamming()
