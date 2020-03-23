"""SPECTRAL-domain processing."""
import math

import numpy as np
from numpy.fft import rfft, irfft


def magphasor(complexspec):
    """Decompose a complex spectrogram into magnitude and unit phasor.

    m, p = magphasor(c) such that c == m * p.
    """
    mspec = np.abs(complexspec)
    pspec = np.empty_like(complexspec)
    zero_mag = (mspec == 0.)  # fix zero-magnitude
    pspec[zero_mag] = 1.
    pspec[~zero_mag] = complexspec[~zero_mag]/mspec[~zero_mag]
    return mspec, pspec


def magphase(cspectrum, unwrap=False):
    """Decompose complex spectrum into magnitude and phase."""
    mag = np.abs(cspectrum)
    phs = np.angle(cspectrum)
    if unwrap:
        phs = np.unwrap(phs)
    return mag, phs


def logmag(sig, floor=-160.):
    """Compute natural log magnitude of complex spectrum.

    Parameters
    ----------
    sig: numpy.ndarray
        Complex spectra.
    floor: float, -80.
        Magnitude floor in dB.

    """
    return np.log(np.abs(sig).clip(10**(floor/20)))


def logpow(sig, floor=-80.):
    """Compute natural log power of complex spectrum.

    Parameters
    ----------
    sig: numpy.ndarray
        Complex spectra.
    floor: float, -80.
        Magnitude floor in dB.

    """
    return np.log((sig.real**2+sig.imag**2).clip(10**(floor/10)))


def phasor(mag, phase):
    """Compute complex spectrum given magnitude and phase."""
    return mag * np.exp(1j*phase)


def dct1(x, dft=False):
    """Perform Type-1 Discrete Cosine Transform (DCT-1) on input signal.

    Parameters
    ----------
    x: array_like
        Signal to be processed.
    dft: boolean
        Implement using dft?

    Returns
    -------
    X: array_like
        Type-1 DCT of x.

    """
    if len(x) == 1:
        return x.copy()
    ndct = len(x)

    if dft:  # implement using dft
        x_ext = np.concatenate((x, x[-2:0:-1]))  # create extended sequence
        X = np.real(rfft(x_ext)[:ndct])
    else:  # implement using definition
        xa = x * 1.
        xa[1:-1] *= 2.  # form x_a sequence
        X = np.zeros_like(xa)
        ns = np.arange(ndct)
        for k in range(ndct):
            cos = np.cos(np.pi*k*ns/(ndct-1))
            X[k] = cos.dot(xa)
    return X


def idct1(x_dct1, dft=False):
    """Perform inverse Type-1 Discrete Cosine Transform (iDCT-1) on spectrum.

    Parameters
    ----------
    x_dct1: array_like
        Input DCT spectrum.
    dft: boolean
        Implement using dft?

    Returns
    -------
    x: array_like
        Inverse Type-1 DCT of x_dct1.

    """
    if len(x_dct1) == 1:
        return x_dct1.copy()
    ndct = len(x_dct1)

    if dft:  # implement using dft
        x = irfft(x_dct1, n=2*(ndct-1))[:ndct]
    else:  # implement using definition
        Xb = x_dct1 / (ndct-1.)
        Xb[0] /= 2.
        Xb[-1] /= 2.
        x = np.zeros_like(Xb)
        ks = np.arange(ndct)
        for n in range(ndct):
            cos = np.cos(np.pi*n*ks/(ndct-1))
            x[n] = cos.dot(Xb)
    return x


def dct2(x, norm=True, dft=False):
    """Perform Type-2 Discrete Cosine Transform (DCT-2) on input signal.

    Parameters
    ----------
    x: array_like
        Input signal.
    norm: boolean, optional
        Normalize so that energy is preserved. Default to True.
    dft: boolean, optional
        Implement using dft? Default to False.

    Returns
    -------
    X: numpy array
        Type-2 DCT of x.

    """
    if len(x) == 1:
        return x.copy()
    ndct = len(x)

    if dft:  # implement using dft
        if norm:
            raise ValueError("DFT method does not support normalization!")
        Xk = rfft(x, 2*ndct)[:ndct]
        X = 2*np.real(Xk*np.exp(-1j*(np.pi*np.arange(ndct)/(2*ndct))))
    else:  # implement using definition
        if norm:
            xa = 1.*x
        else:
            xa = 2.*x
        X = np.zeros_like(xa)
        ns = np.arange(ndct)
        for k in range(ndct):
            cos = np.cos(np.pi*k*(2*ns+1)/(2*ndct))
            X[k] = cos.dot(xa)
            if norm:
                X[k] *= np.sqrt(2./ndct)
        if norm:
            X[0] /= np.sqrt(2)
    return X


def idct2(x_dct2, norm=True, dft=False):
    """Perform inverse Type-2 Discrete Cosine Transform (DCT-2) on spectrum.

    Parameters
    ----------
    x_dct2: array_like
        Input signal.
    norm: boolean, optional
        Normalize so that energy is preserved. Default to True.
    dft: boolean, optional
        Implement using dft? Default to False.

    Returns
    -------
    x: array_like
        Inverse Type-2 DCT of x_dct2.

    """
    if len(x_dct2) == 1:
        return x_dct2.copy()
    ndct = len(x_dct2)

    if dft:  # implement using dft
        if norm:
            raise ValueError("DFT method does not support normalization!")
        ks = np.arange(ndct)
        Xseg1 = x_dct2*np.exp(1j*np.pi*ks/(2*ndct))
        Xseg2 = -x_dct2[-1:0:-1]*np.exp(1j*np.pi*(ks[1:]+ndct)/(2*ndct))
        X_ext = np.concatenate((Xseg1, [0.], Xseg2))
        x = irfft(X_ext[:ndct+1])[:ndct]
    else:  # implement using definition
        if norm:
            Xb = x_dct2 * np.sqrt(2./ndct)
            Xb[0] /= np.sqrt(2.)
        else:
            Xb = x_dct2 / (ndct+0.0)
            Xb[0] /= 2.
        x = np.zeros_like(Xb)
        ks = np.arange(ndct)
        for n in range(ndct):
            cos = np.cos(np.pi*ks*(2*n+1)/(2*ndct))
            x[n] = cos.dot(Xb)
    return x


def mvnorm1(powspec, frameshift, tau=3., tau_init=.1, t_init=.2):
    """Online mean and variance normalization of a short-time power spectra.

    This function computes online mean/variance as a scalar instead of a vector
    in `mvnorm`.

    Parameters
    ----------
    powspec: numpy.ndarray
        Real-valued short-time power spectra with dimension (T,F).
    frameshift: float
        Number of seconds between adjacent frame centers.

    Keyword Parameters
    ------------------
    tau: float, 3.
        Time constant of the median-time recursive averaging function.
    tau_init: float, .1
        Initial time constant for fast adaptation.
    t_init: float, .2
        Amount of time in seconds from the beginning during which `tau_init`
        is applied.
        The rest of time will use `tau`.

    Returns
    -------
    powspec_norm: numpy.ndarray
        Normalized short-time power spectra with dimension (T,F).

    """
    alpha = np.exp(-frameshift / tau)
    alpha0 = np.exp(-frameshift / tau_init)  # fast adaptation
    init_frames = math.ceil(t_init / frameshift)
    assert init_frames < len(powspec)

    mu = np.empty(len(powspec))
    var = np.empty(len(powspec))
    # Start with global mean and variance
    mu[0] = alpha0 * powspec.mean() + (1-alpha0)*powspec[0].mean()
    var[0] = alpha0 * (powspec**2).mean() + (1-alpha0)*(powspec[0]**2).mean()
    for ii in range(1, init_frames):
        mu[ii] = alpha0*mu[ii-1] + (1-alpha0)*powspec[ii].mean()
        var[ii] = alpha0*var[ii-1] + (1-alpha0)*(powspec[ii]**2).mean()
    for ii in range(init_frames, len(powspec)):
        mu[ii] = alpha*mu[ii-1] + (1-alpha)*powspec[ii].mean()
        var[ii] = alpha*var[ii-1] + (1-alpha)*(powspec[ii]**2).mean()

    return (powspec - mu[:, np.newaxis]) / np.maximum(
        np.sqrt(np.maximum(var[:, np.newaxis]-mu[:, np.newaxis]**2, 0)), 1e-12)


def mvnorm(powspec, frameshift, tau=3., tau_init=.1, t_init=.2):
    """Online mean and variance normalization of a short-time power spectra.

    Parameters
    ----------
    powspec: numpy.ndarray
        Real-valued short-time power spectra with dimension (T,F).
    frameshift: float
        Number of seconds between adjacent frames.

    Keyword Parameters
    ------------------
    tau: float, 3.
        Time constant of the median-time recursive averaging function.
    tau_init: float, .1
        Initial time constant for fast adaptation.
    t_init: float, .2
        Amount of time in seconds from the beginning during which `tau_init`
        is applied.
        The rest of time will use `tau`.

    Returns
    -------
    powspec_norm: numpy.ndarray
        Normalized short-time power spectra with dimension (T,F).

    """
    alpha = np.exp(-frameshift / tau)
    alpha0 = np.exp(-frameshift / tau_init)  # fast adaptation
    init_frames = math.ceil(t_init / frameshift)
    assert init_frames < len(powspec)

    mu = np.empty_like(powspec)
    var = np.empty_like(powspec)
    # Start with global mean and variance
    mu[0] = alpha0 * powspec.mean(axis=0) + (1-alpha0)*powspec[0]
    var[0] = alpha0 * (powspec**2).mean(axis=0) + (1-alpha0)*(powspec[0]**2)
    for ii in range(1, init_frames):
        mu[ii] = alpha0*mu[ii-1] + (1-alpha0)*powspec[ii]
        var[ii] = alpha0*var[ii-1] + (1-alpha0)*(powspec[ii]**2)
    for ii in range(init_frames, len(powspec)):
        mu[ii] = alpha*mu[ii-1] + (1-alpha)*powspec[ii]
        var[ii] = alpha*var[ii-1] + (1-alpha)*(powspec[ii]**2)

    return (powspec - mu) / np.maximum(np.sqrt(
        np.maximum(var-mu**2, 0)), 1e-12)
