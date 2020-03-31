"""WINDOW and related utilities for short-time signal analysis."""

import numpy as np
from .util import nextpow2, firfreqz


def hamming(wsize, hop=None, nchan=None, synth=False):
    """Make a hamming window for overlap add analysis and synthesis.

    When `synth` is enabled, the constant overlap-add (COLA) constraint will
    be checked according to FT view (set `hop`) or BPF view (set `nchan`).
    In summary,
    1. Set `synth` if you need perfect synthesis: istft(stft(x)) == x.
    2. Do not set `synth` if you only care about analysis.
    3. Set `hop` if you need COLA in time to unity.
    4. Set `nchan` if you need COLA in frequency to unity.

    Parameters
    ----------
    wsize: int
        Window length in samples.
    hop: float, optional
        Hop fraction in range (0, 1) in FT view.
        Default to None, in which case window is not normalized.
    nchan: int, optional
        Number of frequency channels in BPF view.
        Default to None, in which case window is not normalized.
    synth: bool
        Set if perfect reconstruction is intended.

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
    if synth:
        if (hop is not None):  # for perfect OLA reconstruction in time
            if wsize % 2:  # fix endpoint problem for odd-size window
                wind = np.hamming(wsize)
                wind[0] /= 2.
                wind[-1] /= 2.
            else:  # even-size window
                wind = np.hamming(wsize+1)[:-1]
            assert tnorm(wind, hop),\
                "[wsize:{}, hop:{}] violates COLA in time.".format(wsize, hop)
        elif nchan is not None:  # for perfect filterbank synthesis
            assert fnorm(wind, nchan),\
                "[wsize:{}, nchan:{}] violates COLA in frequency.".format(
                    wsize, nchan)
    else:
        wind = np.hamming(wsize)

    return wind


def rect(wsize, hop=None, nchan=None):
    """Make a rectangular window.

    Parameters
    ----------
    wsize : int
        Window length in samples.
    hop : {None, float}, optional
        Hop fraction in range (0, 1). If unspecified, do not normalize window.

    Returns
    -------
    wind : numpy.ndarray
        A `wsize`-point rectangular window. If `hop` is not None, normalize
        amplitude for constant overlap-add to unity.

    """
    wind = np.ones(wsize)
    if hop is not None:  # for short-time analysis
        tnorm(wind, hop)
    elif nchan is not None:  # for filterbank analysis
        fnorm(wind, nchan)

    return wind


def tnorm(wind, hop):
    """Check COLA constraint before normalizing to OLA unity in time.

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
    tcola : check COLA constraint in time.

    """
    amp = tcola(wind, hop)
    if amp is not None:
        wind /= amp
        return True
    else:
        return False


def fnorm(wind, nchan):
    """Check COLA constraint before normalizing to OLA unity in frequency.

    Parameters
    ----------
    wind : ndarray
        A ``(N,) ndarray`` window function.
    nchan : int
        Number of linear frequency channels.

    Returns
    -------
    success : bool
        True if `wind` and `hop` pass COLA test; `wind` will then be
        normalized in-place. False otherwise; `wind` will be unchanged.

    See Also
    --------
    fcola : check COLA constraint in frequency.

    """
    amp = fcola(wind, nchan)
    if amp is not None:
        wind /= amp
        return True
    else:
        return False


def tcola(wind, hop):
    """Check the constant overlap-add (COLA) constraint in time.

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


def fcola(wind, nchan):
    """Check the constant overlap-add (COLA) constraint in frequency.

    Parameters
    ----------
    wind: array_like
        A `(N,) ndarray` window function.
    nchan: int
        Number of linearly spaced frequency channels in range [0, 2pi).

    Returns
    -------
    amp : float (or None)
        A normalization factor if COLA is satisfied, otherwise None.

    """
    nfft = max(nextpow2(len(wind)), 1024)
    resp = np.zeros(nfft, dtype=np.complex_)
    for kk in range(nchan):
        wk = 2*np.pi * (kk*1./nchan)  # modulation frequency
        _, hh = firfreqz(wind * np.exp(1j*wk*np.arange(len(wind))), nfft)
        resp += hh
    magresp = np.abs(resp)

    if np.allclose(magresp, magresp[0]):
        return magresp[0]
    else:
        return None


def hop2hsize(wind, hop):
    """Interpret hop fraction to integer sample size if necessary."""
    if hop >= 1:
        assert type(hop) == int, "Hop size must be integer!"
        return hop
    else:
        assert 0 < hop < 1, "Hop fraction has to be in range (0,1)!"
        return int(len(wind)*hop)
