# coding: utf-8

"""SPECtral-TEMPoral models for audio signals."""
import math

import numpy as np
from scipy.fftpack import dct, idct
import scipy.signal as signal

from .util import asymfilt, nextpow2
from .temporal import convdn, conv


def ssf(powerspec, lambda_lp, c0=.01, ptype=2):
    """Suppression of Slowly-varying components and the Falling edge.

    This implementation follows paper by Kim and Stern:
    Kim, Chanwoo, and Richard M. Stern."Nonlinear enhancement of onset
    for robust speech recognition." Eleventh Annual Conference of the
    International Speech Communication Association. 2010.

    Parameters
    ----------
    powerspec: numpy.ndarray
        Short-time power spectra. N.B.: This input power spectrum is not
        frequency integrated
    lambda_lp: float
        Time constant to be used as the first-order lowpass filter coefficient.

    Keyword Parameters
    ------------------
    c0: float, 0.01
        Power floor constant.
    ptype: int, 2
        SSF processing type; either 1 or 2.

    Returns
    -------
    out: numpy.ndarray
        If gbank is not specified, this function outputs the ratio
        of processed power to original power (i.e., Eq. (6) in Kim, et al.).
        If gbank is specified, this function outputs the reconstructed
        spectrum (i.e., Eq. (9) in Kim, et al.).

    """

    # Low-pass filtered power
    mspec = signal.lfilter([1-lambda_lp], [1, -lambda_lp], powerspec, axis=0)

    if ptype == 1:
        ptilde = np.maximum(powerspec-mspec, c0*powerspec)
    elif ptype == 2:
        ptilde = np.maximum(powerspec-mspec, c0*mspec)
    else:
        raise ValueError(f"Invalid ptype: [{ptype}]")

    return ptilde / powerspec


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
    mu, _ = signal.lfilter([1-lambda_mu], [1, -lambda_mu], meanpower,
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
    hsa = signal.hilbert(hs, ndft)[:len(hs)]
    hta = signal.hilbert(ht, ndft)[:len(ht)]
    hirs = hs * np.cos(phi) + hsa.imag * np.sin(phi)
    hirt = ht * np.cos(theta) + hta.imag * np.sin(theta)
    hirs_ = signal.hilbert(hirs, ndft)[:len(hs)]
    hirt_ = signal.hilbert(hirt, ndft)[:len(ht)]
    return np.outer(hirt_, hirs_).real,\
        np.outer(np.conj(hirt_), hirs_).real


def strf_gabor(supn, supk, wn, wk):
    """Spectrotemporal receptive fields implemented using the Gabor filters.

    This implementation follows the work of Schadler et al. in
    Schadler, Marc René, Bernd T. Meyer, and Birger Kollmeier. "Spectro-temporal
    modulation subspace-spanning filter bank features for robust automatic
    speech recognition."
    The Journal of the Acoustical Society of America 131.5 (2012): 4134-4151.
    """
    n0 = supn // 2
    k0 = supk // 2
    nspan = np.arange(supn)
    kspan = np.arange(supk)
    nsin = np.exp(1j * wn*(nspan-n0))
    ksin = np.exp(1j * wk*(kspan-k0))
    nwind = .5 - .5 * np.cos(2*np.pi*nspan/(supn+1))
    kwind = .5 - .5 * np.cos(2*np.pi*kspan/(supk+1))
    return np.outer(nsin * nwind, ksin * kwind)


def modspec(sig, sr, fr, fbank, lpf_env, lpf_mod, fc_mod=4, norm=False,
            original=False):
    """Modulation spectrogram proposed by Kingsbury et al.

    Implemented Kingsbury, Brian ED, Nelson Morgan, and Steven Greenberg.
    "Robust speech recognition using the modulation spectrogram."
    Speech communication 25.1-3 (1998): 117-132.

    Parameters
    ----------
    sig: numpy.ndarray
        Time-domain signal to be processed.
    sr, fr: int
        Sampling rate; Frame rate.
    fbank: fbank.Filterbank
        A Filterbank object. .filter() must be implemented.
    """
    assert len(lpf_mod) % 2, "Modulation filter must have odd number of samples."
    ss = len(lpf_mod) // 2
    bpf_mod = lpf_mod * np.exp(1j*2*np.pi*fc_mod/fr * np.arange(-ss, ss+1))
    deci = sr // fr
    nframes = int(math.ceil(len(sig)/deci))
    pspec = np.empty((nframes, len(fbank)))
    if original:
        pspec_orig = np.empty_like(pspec)
    for kk, _ in enumerate(fbank):
        band = fbank.filter(sig, kk)
        if original:
            pspec_orig[:, kk] = band[::deci][:nframes]**2
        banddn = convdn(band.clip(0), lpf_env, deci, True)[:nframes]
        if norm:  # long-term level normalization
            banddn /= banddn.mean()
        banddn = conv(banddn, bpf_mod, True)[:nframes]
        pspec[:, kk] = banddn.real**2 + banddn.imag**2

    if original:
        return pspec, pspec_orig

    return pspec
