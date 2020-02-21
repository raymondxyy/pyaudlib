"""Test spectro-temporal functions."""
import numpy as np
from audlib.quickstart import welcome
from audlib.sig.window import hamming
from audlib.sig.transform import stpowspec
from audlib.sig.fbanks import Gammatone
from audlib.sig.spectemp import strf
from audlib.sig.spectemp import pncc
from audlib.sig.spectemp import ssf
from audlib.enhance import SSFEnhancer


def test_ssf():
    nfft = 1024
    lambda_lp = 1
    number_of_gammatone_filters = 40
    number_of_time_steps = 13
    
    gbank = Gammatone(16000, number_of_gammatone_filters)
    powerspec = np.ones((number_of_time_steps, nfft // 2 + 1))

    # When gbank is given.
    ssf_spectrum = ssf(powerspec, lambda_lp, gbank=gbank, nfft=nfft)
    assert ssf_spectrum.shape[0] == number_of_time_steps, 'returned.shape[0] ({}) does not match {}'.format(ssf_spectrum.shape[0], number_of_time_steps)
    assert ssf_spectrum.shape[1] == nfft // 2 + 1, 'returned.shape[1] ({}) does not match {}'.format(ssf_spectrum.shape[1], nfft // 2 + 1)

    # When gbank is not given.
    ssf_spectrum = ssf(powerspec, lambda_lp)
    assert ssf_spectrum.shape[0] == number_of_time_steps
    assert ssf_spectrum.shape[1] == nfft // 2 + 1

    return


def test_strf():
    frate = 100
    bins_per_octave = 12
    time_support = .2
    freq_support = 1
    phi, theta = 0.5, 0
    kdn, kup = strf(time_support, freq_support, frate, bins_per_octave,
                    phi=phi*np.pi, theta=theta*np.pi)

    return


def test_pncc():
    sig, sr = welcome()
    wlen = .025
    hop = .01
    nfft = 1024
    wind = hamming(int(wlen*sr))
    powerspec = stpowspec(sig, sr, wind, int(hop*sr), nfft, synth=False)
    gtbank = Gammatone(sr, 40)

    wts = gtbank.gammawgt(nfft, powernorm=True, squared=True)
    gammaspec = powerspec @ wts
    g1 = powerspec @ wts

    coef = pncc(gammaspec, tempmask=True)
    for ii in range(10):
        assert np.allclose(g1, gammaspec)
        assert np.allclose(coef, pncc(gammaspec, tempmask=True))
    return coef


if __name__ == "__main__":
    #test_strf()
    test_pncc()
    test_ssf()
