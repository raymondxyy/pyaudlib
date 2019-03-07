"""SPECtral-TEMPoral models for audio signals."""
import numpy as np
from scipy.signal import hilbert


def strf(time, freq, sr, bins_per_octave, rate=1, scale=1, phi=0, theta=0):
    """Spectral-temporal receptive fields for both up and down direction.

    Implement the STRF described in Chi, Ru, and Shamma:
    Chi, T., Ru, P., & Shamma, S. A. (2005). Multiresolution spectrotemporal
    analysis of complex sounds. The Journal of the Acoustical Society of
    America, 118(2), 887–906. https://doi.org/10.1121/1.1945807.

    Parameters
    ----------
    time: int or float
        Time support in seconds. The returned STRF will cover the range
        [0, time).
    freq: int or float
        Frequency support in number of octaves. The returned STRF will
        cover the range [-freq, freq).
    sr: int
        Sampling rate in Hz.
    bins_per_octave: int
        Number of frequency bins per octave on the log-frequency scale.
    rate: int or float
        Stretch factor in time.
    scale: int or float
        Stretch factor in frequency.
    phi: float
        Orientation of spectral evolution in radians.
    theta: float
        Orientation of time evolution in radians.

    """
    def _hs(x, scale):
        """Construct a 1-D spectral impulse response with a 2-diff Gaussian.

        This is the prototype filter suggested by Chi et al.
        """
        sx = scale * x
        return scale * (1-(2**np.pi*sx)**2) * np.exp(-(2*np.pi*sx)**2/2)

    def _ht(t, rate):
        """Construct a 1-D temporal impulse response with a Gamma function.

        This is the prototype filter suggested by Chi et al.
        """
        rt = rate * t
        return rate * rt**2 * np.exp(-3.5*rt) * np.sin(2*np.pi*rt)

    hs = _hs(np.linspace(-freq, freq, endpoint=False,
             num=int(2*freq*bins_per_octave)), scale)
    ht = _ht(np.linspace(0, time, endpoint=False, num=int(sr*time)), rate)
    hsa = hilbert(hs)
    hta = hilbert(ht)
    hirs = hs * np.cos(phi) + hsa.imag * np.sin(phi)
    hirt = ht * np.cos(theta) + hta.imag * np.sin(theta)
    hirs_ = hilbert(hirs)
    hirt_ = hilbert(hirt)

    return np.outer(hirt_, hirs_).real,\
        np.outer(np.conj(hirt_), hirs_).real
