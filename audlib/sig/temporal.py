"""Frame-level time-domain processing."""

import numpy as np
from scipy.linalg import toeplitz, solve_toeplitz, inv
from scipy.signal import fftconvolve


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


def zeroxing(sig, option=None, interp=False):
    """Find all zero-crossing indices in a signal.

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

    return np.sort(np.concatenate((zc_zeros, zc_nonzeros)))


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


def lpc(frame, order, method='autocorr', levinson=False, out='full',
        force_stable=True):
    """Linear predictive coding (LPC).

    Parameters
    ----------
    frame: ndarray
        (Usually windowed) time-domain sequence
    order: int
        LPC order
    method: str [autocorr]
        One of 'autocorr','cov','parcor'
    levinson: bool [False]
        Use Levinson-Durbin recursion? Only available in 'autocorr'.
    out: str [full]
        One of 'full','alpha', where
            full  - [1, -a1, -a2, ..., -ap]
            alpha - [a1, a2, ..., ap]
        'Full' is useful for synthesis; `alpha` is useful to get pole
        locations.

    Returns
    -------
    LPC coefficients as an ndarray.

    """
    assert order < len(frame)
    if method == 'autocorr':  # implement autocorrelation method
        phi = xcorr(frame)
        if levinson:  # use levinson-durbin recursion
            try:
                alpha = solve_toeplitz(phi[:order], phi[1:order+1])
            except np.linalg.linalg.LinAlgError:
                print(
                    "WARNING: singular matrix - adding small value to phi[0]")
                print(phi[:order])
                phi[0] += 1e-9
                alpha = solve_toeplitz(phi[:order], phi[1:order+1])
        else:  # solve by direct inversion.
            alpha = inv(toeplitz(phi[:order])).dot(phi[1:order+1])
        if force_stable and (not lpc_is_stable(alpha)):
            print("Unstable LPC detected. Reflecting back to unit circle.")
            alpha = lpc2stable(alpha)

    elif method == 'cov':  # TODO: implement cov and parcor
        pass
    elif method == 'parcor':
        pass
    else:
        raise ValueError("Method must be one of [autocorr,cov,parcor].")
    if out == 'full':
        return np.insert(-alpha, 0, 1)
    else:
        return alpha


def xcorr(x, y=None, norm=False, biased=True):
    r"""Calculate the cross-correlation between x and y.

    The cross-correlation is defined as:
        \phi_xy[k] = \sum_m x[m]*y[m+k]

    Parameters
    ----------
    x: ndarray
        A time sequence
    y: ndarray [None]
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
        if norm:
            xcf /= xcf[0]
    else:  # cross-correlation mode
        xcf = fftconvolve(x[::-1], y)

    return xcf


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


def ref2lpc(k):
    """Convert a set of reflection coefficients to LPC coefficients.

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
    return lpc2ref(lpc2stable(ref2lpc(k)))


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
