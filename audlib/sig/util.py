# coding: utf-8

"""Utility functions related to audio processing."""

import numpy as np
from numpy.fft import fft
import numpy.random as rand
from scipy.signal import lfilter

FREQZ_CEILING = 1e5


def pre_emphasis(sig, alpha):
    """First-order highpass filter signal."""
    return lfilter([1, -alpha], 1, sig)


def asymfilt(xin, la, lb, zi=None):
    r"""Asymmetric nonlinear filter in eq.4 of Kim and Stern.

    This implementation largely follows paper by Kim and Stern:
    Kim, C., & Stern, R. M. (2016).
    Power-Normalized Cepstral Coefficients (PNCC) for Robust Speech Recognition.
    IEEE/ACM Transactions on Audio Speech and Language Processing, 24(7),
    1315–1329. https://doi.org/10.1109/TASLP.2016.2545928

    Parameters
    ----------
    xin: array_like
        Input signal.
        NOTE: This implementation assumes 2D input for performance. If the
        input is 1D, expand the second axis with xin[:, numpy.newaxis].
    la: float
        Recursive averaging coefficient \lambda_a.
    lb: float
        Recursive averaging coefficient \lambda_b.
    zi: array_like, None
        Initial condition.
        xin.shape[1] == len(zi). Default to all zeros.
    """
    if zi is None:
        zi = np.zeros(xin.shape[1])
    assert xin.shape[1] == len(zi), "Dimension mismatch."
    def filta(qin, qout_tm1): return la * qout_tm1 + (1-la) * qin
    def filtb(qin, qout_tm1): return lb * qout_tm1 + (1-lb) * qin
    xout = np.empty_like(xin)
    mask = xin[0] >= zi
    xout[0, mask] = filta(xin[0, mask], zi[mask])
    xout[0, ~mask] = filtb(xin[0, ~mask], zi[~mask])

    for tt in range(1, len(xin)):
        mask = xin[tt] >= xout[tt-1]
        xout[tt, mask] = filta(xin[tt, mask], xout[tt-1, mask])
        xout[tt, ~mask] = filtb(xin[tt, ~mask], xout[tt-1, ~mask])

    return xout


def dither(sig, norm=False, scale=1e-6):
    """Dither signal by adding small amount of noise to signal.

    Parameters
    ----------
    sig: array_like
        Signal to be processed.
    norm: bool, optional
        Normalize signal amplitude to range [-1, 1] before dithering.
        Default to no.
    scale: float, optional
        Amplitude scale to be applied to Gaussian noise.

    """
    return sig + np.random.randn(*sig.shape)*scale


def clipcenter(sig, threshold):
    """Center clipping by a threshold."""
    if threshold == 0:
        return sig
    out = np.zeros_like(sig)
    threshold = np.abs(threshold)
    maskp = sig > threshold
    maskn = sig < -threshold
    out[maskp] = sig[maskp] - threshold
    out[maskn] = sig[maskn] + threshold
    return out


def clipcenter3lvl(sig, threshold):
    """Three-level center clipping by a threshold."""
    out = np.zeros_like(sig)
    threshold = np.abs(threshold)
    maskp = sig > threshold
    maskn = sig < -threshold
    out[maskp] = 1
    out[maskn] = -1
    return out


def firfreqz(h, nfft):
    """Compute frequency response of an FIR filter."""
    ww = np.linspace(0, 2, num=nfft, endpoint=False)
    hh = fft(h, n=nfft)
    return ww, hh


def iirfreqz(h, nfft, ceiling=FREQZ_CEILING):
    """Compute frequency response of an IIR filter.

    Parameters
    ----------
    h: array_like
        IIR filter coefficent array for denominator polynomial.
        e.g. y[n] = x[n] + a*y[n-1] + b*y[n-2]
             Y(z) = X(z) + a*z^-1*Y(z) + b*z^-2*Y(z)
                                  1
             H(z) = ---------------------------------
                           1 - a*z^-1 -b*z^-2
             h = [1, -a, -b]

    """
    ww = np.linspace(0, 2, num=nfft, endpoint=False)
    hh_inv = fft(h, n=nfft)
    hh = np.zeros_like(hh_inv)
    zeros = hh_inv == 0
    hh[~zeros] = 1 / hh_inv
    hh[zeros] = ceiling
    return ww, hh


def freqz(b, a, nfft, ceiling=FREQZ_CEILING):
    """Compute the frequency response of a z-transform polynomial."""
    ww, hh_numer = firfreqz(b, nfft)
    __, hh_denom = iirfreqz(a, nfft, ceiling=ceiling)
    return ww, hh_numer*hh_denom


def nextpow2(n):
    """Give next power of 2 bigger than n."""
    return 1 << (n-1).bit_length()


def ispow2(n):
    """Check if n is an integer power of 2."""
    return ((n & (n - 1)) == 0) and n != 0


def add_noise(x, n, snr=None):
    """Add user provided noise n with SNR=snr and signal x."""
    noise = additive_noise(x, n, snr=snr)
    if snr == -np.inf:
        return noise
    else:
        return x + noise


def additive_noise(x, n, snr=None):
    """Make additive noise at specific SNR.

    SNR = 10log10(Signal Energy/Noise Energy)
    NE = SE/10**(SNR/10)
    """
    if snr == np.inf:
        return np.zeros_like(x)
    # Modify noise to have equal length as signal
    xlen, nlen = len(x), len(n)
    if xlen > nlen:  # need to append noise several times to cover x range
        nn = np.tile(n, int(np.ceil(xlen/nlen)))[:xlen]
    else:
        nn = n[:xlen]

    if snr == -np.inf:
        return nn
    if snr is None:
        snr = (rand.random()-0.25)*20

    xe = x.dot(x)  # signal energy
    ne = nn.dot(nn)  # noise energy
    nscale = np.sqrt(xe/(10**(snr/10.)) / ne)

    return nscale*nn


def add_white_noise(x, snr=None):
    """Add white noise with SNR=snr to signal x.

    SNR = 10log10(Signal Energy/Noise Energy) = 10log10(SE/var(noise))
    var(noise) = SE/10**(SNR/10)
    """
    n = rand.normal(0, 1, x.shape)
    return add_noise(x, n, snr)


def white_noise(x, snr=None):
    """Return the white noise array given signal and desired SNR."""
    n = rand.normal(0, 1, x.shape)
    if snr is None:
        snr = (rand.random()-0.25)*20
    xe = x.dot(x)  # signal energy
    ne = n.dot(n)  # noise power
    nscale = np.sqrt(xe/(10**(snr/10.)) / ne)  # scaling factor

    return nscale*n


def add_white_noise_rand(x):
    """Add white noise with SNR in range [-10dB,10dB]."""
    return add_white_noise(x, (rand.random()-0.25)*20)


def quantize(x, n):
    """Apply n-bit quantization to signal."""
    x /= np.ma.max(np.abs(x))  # make sure x in [-1,1]
    bins = np.linspace(-1, 1, 2**n+1, endpoint=True)  # [-1,1]
    qvals = (bins[:-1] + bins[1:]) / 2
    bins[-1] = 1.01  # Include 1 in case of clipping
    return qvals[np.digitize(x, bins)-1]
