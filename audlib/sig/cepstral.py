"""Functions for CEPSTRAL processing."""
from functools import reduce
import warnings

import numpy as np
from numpy.fft import rfft, irfft, fft, ifft

from .spectral import magphase, logmag


def clog(cspec, floor=-80.):
    r"""Complex logarithm defined as log(X) = log(|X|)+j\angle(X).
    
    Parameters
    ----------
    cspec: numpy.ndarray
        Complex spectrum.

    Keyword Parameters
    ------------------
    floor: float, -80.
        Floor for log-magnitude in dB.

    """
    mag, phs = magphase(cspec, unwrap=True)
    return logmag(mag, floor) + 1j*phs


def crroots(roots):
    """Convert an iterable of roots to an iterable of complex conjugate pairs.

    Parameters
    ----------
    roots: iterable
        Assume this is returned from numpy.roots. Conjugate pairs must be adjacent.

    Returns
    -------
    root, pair: numpy.complex, bool
        One complex root and if there is a conjugate pair.
    """
    rreal = roots.imag == 0
    assert sum(~rreal) % 2 == 0
    return roots[~rreal][::2], roots[rreal].real

def roots(sig):
    """Find the roots of a finite-duration signal.

    Keyword Parameters
    ------------------
    trim0: bool, False
        Trim zeros at the beginning or the end?

    Returns
    -------
    (rminc, rminr): tuple(numpy.ndarray)
        Complex and real roots inside the unit circle.
        Conjugate pair of each complex root is ignore.
    (rmaxc, rmaxr): tuple(numpy.ndarray)
        Complex and real roots outside the unit circle.
        Conjugate pair of each complex root is ignore.
    gain: float
        sig[0]
    """
    x0 = sig[0]
    roots = np.roots(sig/x0)
    rmag = np.abs(roots)
    assert 1 not in rmag
    rminc, rminr = crroots(roots[rmag < 1])
    rmaxc, rmaxr = crroots(roots[rmag > 1])
    gain = x0 * np.prod(rmaxr) * np.prod([(r.real**2+r.imag**2) for r in rmaxc])
    if len(rmaxr) % 2:  # odd number of zeros outside UC
        gain = -gain
    return (rminc, rminr), (rmaxc, rmaxr), gain


def conjpoly(czeros, rzeros):
    """Compose the polynomial from complex and real zeros.

    Parameters
    ----------
    czeros: numpy.ndarray
        Complex zeros where conjugate pairs are ignored.

    rzeros: numpy.ndarray
        Real zeros.

    Returns
    -------
    Real sequence reprenting the polynomial.

    Note
    ----
    Complex conjugate pairs are assumed and should NOT be passed in. This way
    we make sure the returned sequence is real by following the equation

    (1-az^{-1})(1-a*z^{-1}) = 1 - 2Re{a}z^-1 + |a|^2z^{-2}

    Use numpy.poly if this behavior is not desired.

    """
    cpoly = reduce(np.convolve,
                   ([1, -2*z.real, z.real**2+z.imag**2] for z in czeros))
    rpoly = reduce(np.convolve, ([1, -z] for z in rzeros))
    return np.convolve(cpoly, rpoly)


def ccep_zt(sig, n):
    """Compute complex cepstrum of a signal using the Z-transform method.

    Implementation is based on RS eq 8.68 on page 436.

    Parameters
    ----------
    sig: 1-D numpy.ndarray
        signal to be processed.
    n: int
        index range (-n, n) in which complex cepstrum will be evaluated.

    Returns
    -------
    cep: 1-D ndarray
        complex ceptrum of length `2n-1`; quefrency index (-n, n).

    """
    assert sig[0] != 0, "Leading zero!"
    cep = np.zeros(2*n-1)
    (rminc, rminr), (rmaxc, rmaxr), gain = roots(sig)
    if rmaxc.size > 0:
        for jj, ii in enumerate(range(-n+1, 0)):
            cep[jj] = np.sum(np.abs(rmaxc)**ii*2*np.cos(ii*np.angle(rmaxc)))/ii
    if rmaxr.size > 0:
        for jj, ii in enumerate(range(-n+1, 0)):
            cep[jj] += np.sum(rmaxr**ii)/ii
    cep[n-1] = np.log(np.abs(gain))
    if rminc.size > 0:
        for jj, ii in enumerate(range(1, n)):
            cep[n+jj] = -np.sum(np.abs(rminc)**ii*2*np.cos(ii*np.angle(rminc)))/ii
    if rminr.size > 0:
        for jj, ii in enumerate(range(1, n)):
            cep[n+jj] -= np.sum(rminr**ii)/ii

    return cep


def ccep_dft(sig, n, nfft=4096, floor=-80.):
    """Compute complex cepstrum of short-time signal using the DFT method.

    Parameters
    ----------
    sig: 1-D ndarray
        signal to be processed.
    n: non-negative int
        index range (-n, n) in which complex cepstrum will be evaluated.

    Keyword Parameters
    ------------------
    nfft: int, 4096
        Number of DFT points. Recommended to be large to alleviate time aliasing.
    floor: float, -80.
        log-magnitude floor in dB. Default to -80.

    Returns
    -------
    cep: 1-D ndarray
        complex ceptrum of length `2n-1`; quefrency index (-n, n).

    """
    warnings.warn("Unstable implementation. Use ccep_zt instead.")
    assert n <= nfft//2, "Consider larger nfft!"
    spec = rfft(sig, nfft)
    cep = irfft(clog(spec, floor), nfft)
    return np.concatenate((cep[-(n-1):], cep[:n]))


def rcep_zt(sig, n):
    """Compute real cepstrum using the z-transform method.

    This method computes the complex cepstrum first and take the even part.

    Parameters
    ----------
    sig: 1-D numpy.ndarray
        signal to be processed.
    n: int
        index range [0, n) in which complex cepstrum will be evaluated.

    Returns
    -------
    cep: 1-D ndarray
        complex ceptrum of length `n`; quefrency index [0, n).

    See Also
    --------
    ccep_zt

    """
    ccep = ccep_zt(sig, n)
    rcep = .5*(ccep[n-1:]+ccep[n-1::-1])  # only keep non-negative quefrency
    return rcep


def rcep_dft(sig, n, nfft=4096, floor=-80.):
    """Compute real cepstrum using the DFT method.

    Parameters
    ----------
    sig: 1-D ndarray
        signal to be processed.
    n: int
        Number of cepstrum to return.

    Keyword Parameters
    ------------------
    nfft: int, 4096
        Number of DFT points. Recommended to be large to alleviate time aliasing.
    floor: float, -80.
        Log-magnitude floor in dB.

    Returns
    -------
    cep: 1-D ndarray

    """
    return irfft(logmag(rfft(sig, nfft), floor), nfft)[:n]
