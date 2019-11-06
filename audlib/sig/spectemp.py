"""SPECtral-TEMPoral models for audio signals."""
import numpy as np
from scipy.signal import hilbert, lfilter
from scipy.fftpack import dct, idct

from .util import asymfilt, nextpow2


def pncc(powerspec, medtime=2, medfreq=4, synth=False,
         vad_const=2, lambda_mu=.999, powerlaw=True, cmn=True, ccdim=13,
         tempmask=True, lambda_t=.85, mu_t=.2):
    """Power-Normalized Cepstral Coefficients (PNCC).

    This implementation largely follows paper by Kim and Stern:
    Kim, C., & Stern, R. M. (2016).
    Power-Normalized Cepstral Coefficients (PNCC) for Robust Speech
    Recognition. IEEE/ACM Transactions on Audio Speech and Language Processing,
    24(7), 1315–1329. https://doi.org/10.1109/TASLP.2016.2545928

    Parameters
    ----------

    See Also
    --------
    fbank.Gammatone

    """

    # B. Calculate median-time power
    qtild = np.empty_like(powerspec)
    for mm in range(len(powerspec)):
        ms = max(0, mm-medtime)
        me = min(len(powerspec), mm+medtime+1)
        qtild[mm] = powerspec[ms:me].mean(axis=0)

    # C. Calculate noise floor
    qtild_le = asymfilt(qtild, .999, .5, zi=.9*qtild[0])
    qtild0 = qtild - qtild_le
    qtild0[qtild0 < 0] = 0

    # D. Model temporal masking
    qtild_p = np.empty_like(qtild0)
    qtild_p[0] = qtild0[0]
    for tt in range(1, len(qtild_p)):
        qtild_p[tt] = np.maximum(lambda_t*qtild_p[tt-1], qtild0[tt])

    if tempmask:
        qtild_tm = np.empty_like(qtild0)
        qtild_tm[0] = qtild0[0]
        for tt in range(1, len(qtild_p)):
            mask = qtild0[tt] >= (lambda_t * qtild_p[tt-1])
            qtild_tm[tt, mask] = qtild0[tt, mask]
            qtild_tm[tt, ~mask] = mu_t * qtild_p[tt-1, ~mask]
    else:
        qtild_tm = 0

    # C-D. Track floor of high-passed power envelope
    qtild_f = asymfilt(qtild0, .999, .5, zi=.9*qtild0[0])
    qtild1 = np.maximum(qtild_tm, qtild_f)

    # C-D. Excitation segment vs. non-excitation segment
    excitation = qtild >= vad_const*qtild_le

    # C-D. Compare noise modeling and temporal masking
    rtild = np.empty_like(qtild)
    rtild[excitation] = qtild1[excitation]
    rtild[~excitation] = qtild_f[~excitation]

    # E. Spectral weight smoothing
    stild = np.empty_like(qtild)
    for kk in range(stild.shape[1]):
        ks, ke = max(0, kk-medfreq), min(stild.shape[1], kk+medfreq+1)
        stild[:, kk] = (rtild[:, ks:ke] / qtild[:, ks:ke]).mean(axis=1)

    out = powerspec * stild  # this is T[m,l] in eq.14

    # F. Mean power normalization
    meanpower = out.mean(axis=1)  # T[m]
    mu, _ = lfilter([1-lambda_mu], [1, -lambda_mu], meanpower,
                    zi=[meanpower.mean()])

    if synth:  # return mask only
        return stild / mu[:, np.newaxis]

    out /= mu[:, np.newaxis]  # U[m,l] in eq.16, ignoring the k constant

    # G. Rate-level nonlinearity
    if powerlaw:
        out = out ** (1/15)
    else:
        out = np.log(out + 1e-8)

    # Finally, apply CMN if needed
    out = dct(out, norm='ortho')[:, :ccdim]
    if cmn:
        out -= out.mean(axis=0)

    return out


def pnspec(powerspec, **kwargs):
    """Power spectrum derived from Power-Normalized Cepstral Coefficients.

    See `pncc` for a complete list of function parameters.
    """
    return idct(pncc(powerspec, **kwargs), n=powerspec.shape[1], norm='ortho')


def invspec(tkspec, fkwgts):
    """Invert a short-time spectra or mask with reduced spectral dimensions.

    This is useful when you have a representation like a mel-frequency power
    spectra and want an **approximated** linear frequency power spectra.

    Parameters
    ----------
    tkspec: numpy.ndarray
        T x K short-time power spectra with compressed spectral dim.
    fkwgts: numpy.ndarray
        F x K frequency-weighting matrix used to transform a full power spec.

    Returns
    -------
        T x F inverted short-time spectra.
    """
    return (tkspec @ fkwgts.T) / fkwgts.sum(axis=1)


def strf(time, freq, sr, bins_per_octave, rate=1, scale=1, phi=0, theta=0,
         ndft=None):
    """Spectral-temporal response fields for both up and down direction.

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
        return scale * (1-(2*np.pi*sx)**2) * np.exp(-(2*np.pi*sx)**2/2)

    def _ht(t, rate):
        """Construct a 1-D temporal impulse response with a Gamma function.

        This is the prototype filter suggested by Chi et al.
        """
        rt = rate * t
        return rate * rt**2 * np.exp(-3.5*rt) * np.sin(2*np.pi*rt)

    hs = _hs(np.linspace(-freq, freq, endpoint=False,
             num=int(2*freq*bins_per_octave)), scale)
    ht = _ht(np.linspace(0, time, endpoint=False, num=int(sr*time)), rate)
    if ndft is None:
        ndft = max(512, nextpow2(max(len(hs), len(ht))))
        ndft = max(len(hs), len(ht))
    assert ndft >= max(len(ht), len(hs))
    hsa = hilbert(hs, ndft)[:len(hs)]
    hta = hilbert(ht, ndft)[:len(ht)]
    hirs = hs * np.cos(phi) + hsa.imag * np.sin(phi)
    hirt = ht * np.cos(theta) + hta.imag * np.sin(theta)
    hirs_ = hilbert(hirs, ndft)[:len(hs)]
    hirt_ = hilbert(hirt, ndft)[:len(ht)]
    return np.outer(hirt_, hirs_).real,\
        np.outer(np.conj(hirt_), hirs_).real
