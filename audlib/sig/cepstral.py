"""Functions for CEPSTRAL processing."""
from functools import reduce

import numpy as np
from numpy.fft import rfft, irfft

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


def unwrap(czeros, rzeros, n):
    """Calculate the unwrapped phase from zero locations.

    NOTE: Each complex zero is assumed to implicitly represent a pair of
    zeros in complex conjugate pairs.

    This implementation follows RS Eq. 8.73.

    Parameters
    ----------
    zmaxp: numpy.ndarray
        Zero locations outside the unit circle.
    zminp: numpy.ndarray
        Zero locations inside the unit circle.
    n: int
        DFT points.

    Returns
    -------
    Non-negative frequency phase spectrum in range [0, n//2+1).

    """
    cmags, cphases = czeros
    phase = np.zeros(n//2+1)
    minp = cmags < 1
    for mm, pp in zip(cmags[minp], cphases[minp]):  # minimum-phase
        # (1-az-1)(1-a*z-1) = 1 - 2|a|cos(pp)z-1 + |a|^2 z-2
        phase += np.angle(rfft([1, -2*mm*np.cos(pp*np.pi), mm**2], n))
    for mm, pp in zip(cmags[~minp], cphases[~minp]):  # maximum-phase
        # (1-bz)(1-b*z) = 1 - 2|b|*cos(pp)z + |b|^2 z^2
        ss = np.zeros(n)
        ss[0], ss[-1], ss[-2] = 1, -2/mm*np.cos(pp*np.pi), mm**-2
        phase += np.angle(rfft(ss, n))
    minp = np.abs(rzeros) < 1
    for rr in rzeros[minp]:  # minimum-phase
        phase += np.angle(rfft([1, -rr], n))
    for rr in rzeros[~minp]:  # maximum-phase
        ss = np.zeros(n)
        ss[0], ss[-1] = 1, -1/rr
        phase += np.angle(rfft(ss, n))

    return phase


def roots(sig):
    """Find the roots of a finite-duration real signal.

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
    mr = roots.imag == 0  # real roots
    assert sum(~mr) % 2 == 0
    cr, rr = roots[~mr][::2], roots[mr].real
    rmaxc, rmaxr = cr[np.abs(cr) > 1], rr[np.abs(rr) > 1]
    gain = x0 * np.prod(rmaxr) * np.prod([(r.real**2+r.imag**2) for r in rmaxc])
    if len(rmaxr) % 2:  # odd number of zeros outside UC
        gain = -gain

    return (np.abs(cr), np.angle(cr)/np.pi), rr, gain


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
    czeros, rzeros, gain = roots(sig)
    return z2ccep(czeros, rzeros, gain, n)


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
    compphase: bool, True.
        Compensate linear phase term by finding the number of roots of the
        polynomial representing the z transform of the signal that are outside
        the unit circle (maximum-phase zeros).

    Returns
    -------
    cep: 1-D ndarray
        complex ceptrum of length `2n-1`; quefrency index (-n, n).

    """
    assert n <= nfft//2, "Consider larger nfft!"
    spec = rfft(sig, nfft)
    czeros, rzeros, _ = roots(sig)
    cep = irfft(logmag(spec, floor)+1j*unwrap(czeros, rzeros, nfft), nfft)
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


def z2ccep(czeros, rzeros, gain, n):
    """Convert zero locations to complex cepstrum.

    Parameters
    ----------
    zeros: iterable
        Zeros locations in an array of amplitude,phase pair.
        NOTE: complex zeros are assumed to appear in conjugate pairs, so only
        one zero of each pair should be passed in.
    n: int
        Index range (-n, n) in which complex cepstrum will be evaluated.

    """
    cep = np.zeros(2*n-1)
    cmags, cphases = czeros
    cmax = cmags > 1 # maxphase complex zeros
    if cmax.sum() > 0:
        rr, pp = cmags[cmax], cphases[cmax]
        for jj, ii in enumerate(range(-n+1, 0)):
            cep[jj] = np.sum(rr**ii*2*np.cos(ii*pp*np.pi))/ii
    rmax = rzeros > 1  # maxphase real zeros
    if rmax.sum() > 0:
        for jj, ii in enumerate(range(-n+1, 0)):
            cep[jj] += np.sum(rzeros[rmax]**ii)/ii
    cep[n-1] = np.log(np.abs(gain))
    if (~cmax).sum() > 0:  # minphase complex zeros
        rr, pp = cmags[~cmax], cphases[~cmax]
        for jj, ii in enumerate(range(1, n)):
            cep[n+jj] = -np.sum(rr**ii*2*np.cos(ii*pp*np.pi))/ii
    if (~rmax).sum() > 0:  # minphase real zeros
        for jj, ii in enumerate(range(1, n)):
            cep[n+jj] -= np.sum(rzeros[~rmax]**ii)/ii

    return cep


def p2ccep(cpoles, rpoles, gain, n):
    """Convert pole locations to complex cepstrum.

    Parameters
    ----------
    poles: array_like
        Zeros locations in an array.
        NOTE: complex zeros are assumed to appear in conjugate pairs, so only
        one zero of each pair should be passed in.
    n: int
        Index range (-n, n) in which complex cepstrum will be evaluated.

    """
    cep = np.zeros(n)
    cep[0] = np.log(np.abs(gain))
    cmags, cphases = cpoles
    if len(cmags) > 0:  # minimum-phase complex poles
        for ii in range(1, n):
            cep[ii] = np.sum(cmags**ii*2*np.cos(ii*np.pi*cphases))/ii
    if len(rpoles) > 0:  # minimum-phase real poles
        for ii in range(1, n):
            cep[ii] += np.sum(rpoles**ii)/ii

    return cep


def zp2ccep(czeros, rzeros, cpoles, rpoles, gain, n):
    """Convert zero and pole locations to complex cepstrum."""
    cep = z2ccep(czeros, rzeros, gain, n)
    cep[n-1:] = cep[n-1:] + p2ccep(cpoles, rpoles, 1, n)
    return cep
