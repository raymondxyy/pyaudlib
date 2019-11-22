"""Test suite for audlib.sig.cepstral."""
import numpy as np

from audlib.quickstart import welcome
from audlib.sig.cepstral import rcep_zt, rcep_dft, ccep_zt, ccep_dft


def test_cceps():
    """Test complex cepstrum function using the impulse response of an echo.

    Taken from example 8.3 of RS, page 414.

    """
    # the complex cepstrum of simple echo is an impulse train with decaying
    # magnitude.
    Np = 8
    alpha = .5  # echo impulse
    x = np.zeros(512)
    x[0], x[Np] = 1, alpha

    cepsize = 150
    # construct reference
    cepref = np.zeros(2*cepsize-1)
    deltalocs = np.zeros_like(cepref, dtype='bool')
    deltalocs[cepsize-1+Np::Np] = True
    cepref[deltalocs] = np.array( # first delta at x[Np]
        [(-1)**(k+1) * (alpha**k)/k for k in range(
            1, cepref[deltalocs].size+1)]
    )

    # test complex cepstrum
    ccep1 = ccep_zt(x, cepsize)
    ccep2 = ccep_dft(x, cepsize)
    assert np.allclose(cepref, ccep1)
    assert np.allclose(cepref, ccep2, atol=1e-6)

    # test real cepstrum
    cepref = .5*(cepref + cepref[::-1])[cepsize-1:]
    rcep1 = rcep_zt(x, cepsize)
    rcep2 = rcep_dft(x, cepsize)
    assert np.allclose(cepref, rcep1)
    assert np.allclose(cepref, rcep2)


def test_rceps():
    """Test audlib.sig.cepstral.rcep*"""
    sig, _ = welcome()
    eesig = sig[8466:8466+512]
    nceps = 200
    rcep1 = rcep_zt(eesig, nceps)
    rcep2 = rcep_dft(eesig, nceps)
    # Give a relatively high tolerance because of time-aliasing
    assert np.allclose(((rcep1-rcep2)**2).mean(), [0], atol=1e-6)


if __name__ == "__main__":
    test_rceps()
    test_cceps()
