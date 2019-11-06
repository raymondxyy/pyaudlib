"""Test spectro-temporal functions."""
import numpy as np
from audlib.sig.spectemp import strf
from audlib.quickstart import welcome
from audlib.sig.window import hamming
from audlib.sig.transform import stpowspec, stft, istft
from audlib.sig.fbanks import Gammatone
from audlib.sig.spectemp import pncc, invspec


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


def test_invpnspec():
    # TODO
    sig, sr = welcome()
    wlen = .025
    hop = .01
    nfft = 1024
    wind = hamming(int(wlen*sr))
    spec = stft(sig, sr, wind, hop, nfft, synth=True, zphase=True)
    pspec = spec.real**2 + spec.imag**2
    gtbank = Gammatone(sr, 40)

    wts = gtbank.gammawgt(nfft, powernorm=True, squared=True)
    gammaspec = pspec @ wts

    mask40 = pncc(gammaspec, tempmask=True, synth=True)
    maskfull = invspec(mask40, wts)
    sigsynth = istft((maskfull**.5)*spec, sr, wind, hop, nfft, zphase=True)

    return


if __name__ == "__main__":
    #test_strf()
    test_invpnspec()
