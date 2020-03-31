"""Test spectro-temporal functions."""
import numpy as np
from audlib.quickstart import welcome
from audlib.sig.window import hamming
from audlib.sig.transform import stpowspec, stft, istft
from audlib.sig.fbanks import Gammatone
from audlib.sig.spectemp import strf, pncc, ssf, invspec


WELCOME, SR = welcome()


def test_ssf():
    powerspec = stpowspec(WELCOME, hamming(int(.025*SR)), .5, 512)
    wts = Gammatone(SR, 40).gammawgt(512)

    mask = ssf(powerspec @ wts, .4)
    assert mask.shape[0] == powerspec.shape[0]
    assert mask.shape[1] == wts.shape[1]


def test_strf():
    frate = 100
    bins_per_octave = 12
    time_support = .2
    freq_support = 1
    phi, theta = 0.5, 0
    kdn, kup = strf(time_support, freq_support, frate, bins_per_octave,
                    phi=phi*np.pi, theta=theta*np.pi)
    return


def test_strf_gabor():
    # TODO: real test case
    from audlib.sig.spectemp import strf_gabor
    gabor = strf_gabor(40, 20, 0.2*np.pi, .3*np.pi)
    return


def test_pncc():
    wlen = .025
    hop = .01
    nfft = 1024
    wind = hamming(int(wlen*SR))
    powerspec = stpowspec(WELCOME, wind, int(hop*SR), nfft, synth=False)
    gtbank = Gammatone(SR, 40)

    wts = gtbank.gammawgt(nfft, powernorm=True, squared=True)
    gammaspec = powerspec @ wts
    g1 = powerspec @ wts

    coef = pncc(gammaspec, tempmask=True)
    for ii in range(10):
        assert np.allclose(g1, gammaspec)
        assert np.allclose(coef, pncc(gammaspec, tempmask=True))
    return coef


def test_invpnspec():
    # TODO
    wlen = .025
    hop = .01
    nfft = 1024
    wind = hamming(int(wlen*SR))
    spec = stft(WELCOME, wind, hop, nfft, synth=True, zphase=True)
    pspec = spec.real**2 + spec.imag**2
    gtbank = Gammatone(SR, 40)

    wts = gtbank.gammawgt(nfft, powernorm=True, squared=True)
    gammaspec = pspec @ wts

    mask40 = pncc(gammaspec, tempmask=True, synth=True)
    maskfull = invspec(mask40, wts)
    sigsynth = istft((maskfull**.5)*spec, wind, hop, nfft, zphase=True)

    return


if __name__ == "__main__":
    #test_strf()
    #test_pncc()
    #test_ssf()
    #test_invpnspec()
    test_strf_gabor()
