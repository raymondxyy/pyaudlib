"""WINDOW and related utilities for short-time signal analysis."""

import numpy as np


def hamming(wsize, hop=None, normdc=False):
    """Make a hamming window for overlap add analysis and synthesis.

    The end points of the traditional hamming window are fixed to produce COLA
    window. If you want the original hamming window, use `numpy.hamming`.

    Parameters
    ----------
    wsize : int
        Window length in samples.
    hop : {None, float}, optional
        Hop fraction in range (0, 1). If unspecified, do not normalize window.

    Returns
    -------
    wind : ndarray
        A `wsize`-point hamming window. If `hop` is not None, normalize
        amplitude for constant overlap-add to unity.

    References
    ----------
    Fixed end-point issues for COLA following Julius' Smith's code.

    See Also
    --------
    normalize : Normalize window amplitude for unity overlap-add.

    """
    if wsize % 2:  # fix endpoint problem for odd-size window
        wind = np.hamming(wsize)
        wind[0] /= 2.
        wind[-1] /= 2.
    else:  # even-size window
        wind = np.hamming(wsize+1)
        wind = wind[:-1]
    if hop is not None:
        normalize(wind, hop)
    elif normdc is not None:
        wind /= wind.sum()
    return wind


def rect(wsize, hop=None, normdc=False):
    """Make a rectangular window.

    Parameters
    ----------
    wsize : int
        Window length in samples.
    hop : {None, float}, optional
        Hop fraction in range (0, 1). If unspecified, do not normalize window.

    Returns
    -------
    wind : ndarray
        A `wsize`-point rectangular window. If `hop` is not None, normalize
        amplitude for constant overlap-add to unity.

    See Also
    --------
    normalize : Normalize window amplitude for unity overlap-add.

    """
    wind = np.ones(wsize)
    if hop is not None:  # for short-time analysis
        normalize(wind, hop)
    elif normdc is not None:  # for filterbank analysis
        wind /= wsize

    return wind


def normalize(wind, hop):
    """Check COLA constraint before normalizing window to OLA unity.

    Parameters
    ----------
    wind : ndarray
        A ``(N,) ndarray`` window function.
    hop : float
        Hop fraction in range (0, 1).

    Returns
    -------
    success : bool
        True if `wind` and `hop` pass COLA test; `wind` will then be
        normalized in-place. False otherwise; `wind` will be unchanged.

    See Also
    --------
    cola : check COLA constraint.
    """

    amp = cola(wind, hop)
    if amp is not None:
        wind /= amp
        return True
    else:
        print("WARNING: wind, hop does not conform to COLA.")
        return False


def cola(wind, hop):
    """Check the constant overlap-add (COLA) constraint.

    Parameters
    ----------
    wind : ndarray
        A ``(N,) ndarray`` window function.
    hop : float
        Hop fraction in range (0, 1).

    Returns
    -------
    amp : float (or None)
        A normalization factor if COLA is satisfied, otherwise None.
    """
    wsize = len(wind)
    hsize = hop2hsize(wind, hop)
    buff = wind.copy()  # holds OLA buffer and account for time=0
    for wi in range(hsize, wsize, hsize):  # window moving forward
        wj = wi+wsize
        buff[wi:] += wind[:wsize-wi]
    for wj in range(wsize-hsize, 0, -hsize):  # window moving backward
        wi = wj-wsize
        buff[:wj] += wind[wsize-wj:]

    if np.allclose(buff, buff[0]):
        return buff[0]
    else:
        return None


def hop2hsize(wind, hop):
    """Convert hop fraction to integer size if necessary."""
    if hop >= 1:
        assert type(hop) == int, "Hop size must be integer!"
        return hop
    else:
        assert 0 < hop < 1, "Hop fraction has to be in range (0,1)!"
        return int(len(wind)*hop)
