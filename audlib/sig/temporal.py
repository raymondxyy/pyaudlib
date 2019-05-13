"""Frame-level time-domain processing."""
import math
import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.linalg import toeplitz, solve_toeplitz, inv, cholesky
from scipy.signal import fftconvolve, lfilter

from .util import freqz


def conv(sig, hr, zphase=False):
    """Linear convolution.

    This is at the moment a simple wrapper of numpy.convolve.

    Parameters
    ----------
    sig: array_like
        Signal to be processed.
    hr: array_like
        Impulse response of the filter.
    zphase: bool
        Assume `hr` is centered at time 0? Default to False.

    """
    if zphase:
        return np.convolve(sig, hr)[(len(hr)-1)//2:]
    else:
        return np.convolve(sig, hr)


def convdn(sig, hr, decimate, zphase=False):
    """Efficient implementation of convolution followed by downsampling.

    Parameters
    ----------
    sig: array_like
        Signal to be processed.
    hr: array_like
        Impulse response of the filter.
    decimate: int
        Decimation factor.
        Note that no lowpass filtering is done before decimation.
    zphase: bool, optional
        Assume `hr` is centered at time 0? Default to False.

    """
    if len(sig) < len(hr):
        sig, hr = hr, sig
    hsize = len(hr)
    ssize = len(sig)
    if zphase:
        osize = math.ceil((ssize + (hsize - 1)//2)/decimate)
    else:
        osize = math.ceil((ssize + hsize - 1)/decimate)
    # Calculate zero-padding sizes
    zpleft = (hsize-1)//2 if zphase else hsize-1
    sigpad = np.zeros((osize-1)*decimate+hsize)
    zpright = len(sigpad) - zpleft - ssize
    if zpright < 0:  # happens when decimation factor becomes large
        sigpad[zpleft:] = sig[:len(sigpad)-zpleft]
    else:
        sigpad[zpleft:len(sigpad)-zpright] = sig

    std = sig.strides[0]
    buf = as_strided(sigpad, shape=(osize, hsize), strides=(std*decimate, std))

    return buf.dot(hr[::-1])


def convup(sig, h, interp):
    """Efficient implementation of upsampling followed by convolution.

    Parameters
    ----------
    sig: array_like
        Signal to be processed.
    h: array_like
        Impulse response of the filter.
    interp: int
        Interpolation factor.

    """
    raise NotImplementedError


def zcpa(sig, sr, option=None, interp=False, nyquist=True):
    """Zero-crossing Peak Amplitude.

    Implementation according to:
    [Kim1996](https://doi.org/10.1109/ICASSP.1996.540290)

    Parameters
    ----------
    sig: array_like
        Signal to be processed.
    option: str, optional
        'up' for upward zero-crossings; 'down' for downward zero-crossings.
        Default to None, which counts both types.
    interp: bool
        If True, do linear interpolation to find the fraction zero-crossing.
        Otherwise return integers.

    """
    zc = zeroxing(sig, option, interp)
    freq = sr*1. / (zc[1:]-zc[:-1])
    peak_amp = np.zeros_like(zc[:-1])
    for ii, (n1, n2) in enumerate(zip(zc[:-1], zc[1:])):
        peak_amp[ii] = np.abs(sig[int(n1):int(n2)]).max()

    if nyquist:
        freq_under_nyquist = freq <= (sr/2.)
        freq = freq[freq_under_nyquist]
        peak_amp = peak_amp[freq_under_nyquist]

    return freq, peak_amp


def zeroxing(sig, option=None, sort=True, interp=False):
    """Find all zero-crossing indices in a signal.

    Parameters
    ----------
    sig: array_like
        Signal to be processed.
    option: str, optional
        'up' for upward zero-crossings; 'down' for downward zero-crossings.
        Default to None, which counts both types.
    sort: bool, optional
        Sort the zero-crossings by index. Default to true.
    interp: bool
        If True, do linear interpolation to find the fraction zero-crossing.
        Otherwise return integers.

    """
    signs = np.sign(sig)
    zeros = signs[:-1] == 0
    if option == 'up':  # upward only
        zc_zeros = zeros & (signs[1:] == 1) & (
            np.insert(signs[:-2], 0, 0) == -1)
        zc_nonzeros = ~zeros & (signs[:-1] == -1) & (signs[1:] == 1)
    elif option == 'down':  # downward only
        zc_zeros = zeros & (signs[1:] == -1) & (
            np.insert(signs[:-2], 0, 0) == 1)
        zc_nonzeros = ~zeros & (signs[:-1] == 1) & (signs[1:] == -1)
    else:  # upward and downward
        zc_zeros = zeros & (signs[1:] != 0) & (
            np.insert(signs[:-2], 0, 0) != 0)
        zc_nonzeros = ~zeros & ((signs[:-1] == 1) & (signs[1:] == -1)
                                | (signs[:-1] == -1) & (signs[1:] == 1))

    # Find actual indices instead of booleans
    zc_zeros, = np.where(zc_zeros)
    zc_nonzeros, = np.where(zc_nonzeros)

    if interp:  # linear interpolate nonzero indices
        mag_n = np.abs(sig[zc_nonzeros])
        mag_np1 = np.abs(sig[zc_nonzeros+1])
        zc_nonzeros = (mag_n*(zc_nonzeros+1) + mag_np1*zc_nonzeros)\
            / (mag_n + mag_np1)

    res = np.concatenate((zc_zeros, zc_nonzeros))
    if sort:
        res = np.sort(res)

    return res


def zcrate(sig, option=None):
    """Compute the zero-crossing rate of a signal.

    Parameters
    ----------
    sig: array_like
        Signal to be processed.
    option: str, optional
        'up' for upward zero-crossings; 'down' for downward zero-crossings.
        Default to None, which counts both types.

    Returns
    -------
    out: float in range [0, 1]
        Number of zero-crossings / signal length.

    See Also
    --------
    zeroxing

    """
    return len(zeroxing(sig, option=option)) / len(sig)


def xcorr(x, y=None, norm=False, biased=True):
    r"""Calculate the cross-correlation between x and y.

    The cross-correlation is defined as:
        \phi_xy[k] = \sum_m x[m]*y[m+k]

    Parameters
    ----------
    x: ndarray
        A time sequence
    y: ndarray, optional
        Another time sequence; default to x if None.
    one_side: bool
        Returns one-sided correlation sequence starting at index 0 if
        True, otherwise returns the full sequence. This is only useful
        in the case where y is None.
    norm: bool
        If true, divide the entire function by acf[0]/ccf[0].
    biased: bool
        If false, scale the entire function by 1/(N-m).

    Returns
    -------
    The cross-correlation sequence

    """
    if y is None:  # auto-correlation mode
        xcf = fftconvolve(x[::-1], x)[len(x)-1:]
        if not biased:
            xcf /= np.arange(len(xcf), 0, -1)
        if norm and (xcf[0] > 1e-8):  # prevent zero division in silent frames
            xcf /= xcf[0]
    else:  # cross-correlation mode
        xcf = fftconvolve(x[::-1], y)

    return xcf


def lpcerr(sig, alphas, gain=None):
    """Compute the error signal using LPC coefficents."""
    a = gain if gain is not None else 1
    b = pred2poly(alphas)
    return lfilter(b, a, sig)


def lpcspec(alphas, nfft, gain=None):
    """Compute magnitude spectrum envelope using LPC coefficents."""
    b = gain if gain is not None else 1
    a = pred2poly(alphas)
    ww, hh = freqz(b, a, nfft)
    return ww, np.abs(hh)


def lpc_atc(sig, order, levinson=True, stable=True):
    """Linear predictive coding using the autocorrelation method.

    Parameters
    ----------
    sig: array_like
        (Usually windowed) time-domain sequence.
    order: int
        LPC order.
    levinson: bool, optional
        Use Levinson-Durbin recursion? Default to True.
    stable: bool, optional
        Enforce stability for pole locations? Default to True.

    Returns
    -------
    alphas: numpy.ndarray
        `order`-point LPC coefficients: [a1, a2, ..., ap].
        The all-pole filter can be reconstructed from the diff eq:
            y[n] = G*x[n] + a1*y[n-1] + a2*y[n-2] + ... + ap*y[n-p]
    gain: float
        Filter gain.

    """
    rxx = xcorr(sig)
    if levinson:  # use levinson-durbin recursion
        try:
            alphas = solve_toeplitz(rxx[:order], rxx[1:order+1])
        except np.linalg.linalg.LinAlgError:
            print("Singular matrix!! Adding small value to phi[0].")
            print(rxx[:order])
            rxx[0] += 1e-9
            alphas = solve_toeplitz(rxx[:order], rxx[1:order+1])
    else:  # solve by direct inversion.
        alphas = inv(toeplitz(rxx[:order])).dot(rxx[1:order+1])
    if stable and (not lpc_is_stable(alphas)):
        print("Unstable LPC detected!! Reflecting back to unit circle.")
        alphas = lpc2stable(alphas)

    gain = np.sqrt(rxx[0] - rxx[1:order+1].dot(alphas))

    return alphas, gain


def lpc_cov(sig, order):
    """Linear predictive coding using the covariance method."""
    rxx = np.empty((order+1, order+1))
    sn1 = np.empty_like(sig)
    sn2 = np.empty_like(sig)
    slen = len(sig)
    for kk in range(order+1):
        sn2[kk:slen] = sig[:slen-kk]
        for ii in range(order+1):
            sn1[ii:slen] = sig[:slen-ii]
            rxx[ii, kk] = sn1.dot(sn2)
            sn1[:] = 0
        sn2[:] = 0
    rinv = inv(cholesky(rxx[1:, 1:]))
    alphas = (rinv @ rinv.T) @ rxx[1:, 0]
    gain = np.sqrt(rxx[0, 0] - rxx[0, 1:order+1].dot(alphas))

    return alphas, gain


def lpc_parcor(sig, order):
    """Linear predictive coding using the PARCOR method."""
    slen = len(sig)
    bi = np.zeros(slen + order)
    ei = np.zeros_like(bi)
    ei[:slen] = sig
    bi[order:] = sig
    ks = np.empty(order)  # holds reflection coefficients
    mse = sig.dot(sig)
    for ii in range(1, order+1):
        etmp = ei[:slen+ii]
        btmp = bi[order-ii:]
        ks[ii-1] = etmp.dot(btmp) / np.sqrt(ei.dot(ei)*bi.dot(bi))
        ei[:slen+ii], bi[order-ii:] = \
            etmp - ks[ii-1] * btmp, btmp - ks[ii-1] * etmp
        mse *= (1-ks[ii-1]**2)

    gain = np.sqrt(mse)

    return ks, gain


def lpc2ref(alpha):
    """Convert a set of LPC alphas to reflection coefficients.

    Parameters
    ----------
    alpha: ndarray
        LPC coefficients (excluding 1)

    Returns
    -------
    k: ndarray
        Reflection coefficients of the same order as alpha.

    """
    order = len(alpha)
    a = np.zeros((order, order))
    a[-1] = alpha
    for i in range(order-2, -1, -1):
        a[i, :i+1] = (a[i+1, :i+1]+a[i+1, i+1] * np.flipud(a[i+1, :i+1]))\
                   / (1-a[i+1, i+1]**2)
    return np.diag(a)


def ref2pred(k):
    """Convert a set of reflection coefficients to prediction coefficients.

    Parameters
    ----------
    k: ndarray
        reflection coefficients

    Returns
    -------
    alpha: ndarray
        LPC coefficients (excluding 1)

    """
    alphas = np.diag(k)
    for i in range(1, alphas.shape[0]):
        alphas[i, :i] = alphas[i-1, :i] - k[i]*np.flipud(alphas[i-1, :i])
    return alphas[-1]


def lpc2stable(alpha):
    """Reflect any pole location outside the unit circle inside.

    Parameters
    ----------
    alpha: ndarray
        LPC coefficients

    Returns
    -------
    Stable LPC coefficients

    """
    poles = np.roots(np.insert(-alpha, 0, 1))
    for i in range(len(poles)):
        if np.abs(poles[i]) > 1:
            poles[i] /= (np.abs(poles[i])**2)  # reflect back to unit circle
        if np.abs(poles[i]) > (1/1.01):
            # FIXME: this is a temporary fix for pole location very close to 1
            # it might cause instability after ld recursion
            poles[i] /= np.abs(poles[i])
            poles[i] /= 1.01
    alpha_s = -np.poly(poles)[1:]
    if not lpc_is_stable(alpha_s):
        raise ValueError("`lpc2stable` does not work!")
    return alpha_s


def ref2stable(k):
    """Make reflection coefficients stable."""
    return lpc2ref(lpc2stable(ref2pred(k)))


def ref_is_stable(k):
    """Check if the set of reflection coefficients is stable."""
    return np.all(np.abs(k) < 1)


def lpc_is_stable(alpha):
    """Check if the set of LPC coefficients is stable."""
    return ref_is_stable(lpc2ref(alpha))


def ref2lar(k):
    """Convert a set of reflection coefficients to log area ratio.

    Parameters
    ----------
    k: ndarray
        reflection coefficients
    Returns
    -------
    g: ndarray
        log area ratio (lar)

    """
    if np.greater_equal(k, 1).any():
        raise ValueError(
            "Reflection coefficient magnitude must be smaller than 1.")
    try:
        lar = np.log((1-k))-np.log((1+k))
    except RuntimeWarning:
        print("Invalid log argument")
        print(k)
        lar = 0
    return lar


def lpc2lar(alpha):
    """Convert a set of LPC coefficients to log area ratio."""
    return ref2lar(lpc2ref(alpha))


def pred2poly(alphas):
    """Convert a set of LPC coefficients to polynomial coefficents."""
    b = np.empty(len(alphas)+1)
    b[0] = 1
    b[1:] = -alphas
    return b
