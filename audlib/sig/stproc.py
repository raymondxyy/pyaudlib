"""Short-Time PROCessing of audio signals.

This module implements common audio analysis and synthesis techniques based on
short-time signal analysis. All short-time representations are returned as
GeneratorType.

See Also
--------
fbank : Filterbank analysis and synthesis

Examples
--------

"""

from types import GeneratorType

import numpy as np
from numpy.fft import rfft, irfft
from scipy.linalg import toeplitz, solve_toeplitz, inv
from scipy.signal import fftconvolve

from .window import hop2hsize


def stana(sig, sr, wind, hop, trange=(None, None)):
    """[S]hort-[t]ime [Ana]lysis of audio signal.

    Analyze a audio/speech-like time series by windowing. Yield each frame on
    demand.

    Arguments
    ---------
    sig: 1-D ndarray
        time series to be analyzed.
    sr: int
        sampling rate.
    wind: 1-D ndarray
        window function used for framing. See `window.py` for window functions.
    hop: int, float
        hop size or fraction.

    Keyword Arguments
    -----------------
    trange: tuple [(None, None)]
        time range of `sig` to be analyzed.
        NOTE: default (None) corresponds to analyzing the time range
        (int(-len(wind) * (1-hop_fraction)), len(sig))
        which satisfies COLA (see window.py) of entire signal duration during
        synthesis. Make appropriate change (e.g. (0,None)) to cover specific
        time range in seconds.

    Outputs
    -------
    A generator. Each iteration yields a windowed signal frame.
    """
    ssize = len(sig)
    fsize = len(wind)
    hsize = hop2hsize(wind, hop)
    hfrac = hsize*1. / fsize
    sstart, send = trange
    if sstart is None:
        sstart = int(-fsize * (1-hfrac))
    else:
        sstart = int(sstart * sr)
    if send is None:
        send = ssize
    else:
        send = int(send * sr)

    for si in range(sstart, send, hsize):
        sframe = np.zeros(fsize)  # frame buffer to be yielded
        sj = si + fsize
        if sj <= 0:  # entire frame 0s
            continue
        else:
            if si < 0:  # [0 0 ... x[0] x[1] ...]
                sframe[-si:] = sig[:sj]
            elif sj > ssize:  # [... x[-2] x[-1] 0 0 ... 0]
                sframe[:ssize-si] = sig[si:]
            else:  # [x[..] ..... x[..]]
                sframe[:] = sig[si:sj]

        yield sframe*wind


def ola(sframes, sr, wind, hop):
    """Short-time Synthesis by [O]ver[l]ap-[A]dd.

    Perform the Overlap-Add algorithm on an iterable of short-time analyzed
    frames. Arguments used to call `stana` should be used here for consistency.

    Arguments
    ---------
    sframes: iterable of 1-D ndarray
        array of short-time frames (see `stana`).
    sr: int
        sampling rate
    wind: 1-D ndarray
        window function (see `pyaudiolib.sig.window`).
    hop: int, float
        hop size or hop fraction of window.

    Outputs
    -------
    sout: 1-D ndarray
        Reconstructed time series starting at `sig[0]` assuming `sig` is the
        time series.
    """
    if isinstance(sframes, GeneratorType):
        sframes = np.array(list(sframes))  # big memory footprint..
    nframe = len(sframes)
    fsize = len(wind)
    hsize = hop2hsize(wind, hop)
    hfrac = hsize*1. / fsize
    # Assume analysis is done properly, with pre-appended 0s that satisfies
    # COLA throughout nonzero values in the sequence.
    ssize = hsize*(nframe-1)+fsize  # total OLA size
    sstart = int(-fsize * (1-hfrac))  # OLA starting index
    send = ssize + sstart  # OLA ending index
    ii = sstart  # pointer to beginning of current time frame

    sout = np.zeros(send)
    for frame in sframes:
        frame = frame[:fsize]  # for cases like DFT
        if ii < 0:  # first (few) frames
            sout[:ii+fsize] += frame[-ii:]
        elif ii+fsize > send:  # last (few) frame
            sout[ii:] += frame[:(send-ii)]
        else:
            sout[ii:ii+fsize] += frame
        ii += hsize

    return sout


def stft(sig, sr, wind, hop, nfft, trange=(None, None), zphase=True):
    """Short-time Fourier Transform.

    Implement STFT with the Fourier transform view. See `stana` for meanings
    of short-time analysis arguments.

    Arguments
    ---------
    nfft: int
        DFT size.

    Keyword Arguments
    -----------------
    zphase: bool [True]
        implement zero-phase STFT?

    Outputs
    -------
    sout: iterable of 1-D ndarray
        iterable of STFT frames.
    """
    fsize = len(wind)
    woff = (fsize-(fsize % 2)) // 2
    zp = np.zeros(nfft-fsize)  # zero padding
    for frame in stana(sig, sr, wind, hop, trange):
        if zphase:
            frame = np.concatenate((frame[woff:], zp, frame[:woff]))
        else:  # conventional linear-phase STFT
            frame = np.concatenate((frame, zp))
        yield rfft(frame)


def istft(sframes, sr, wind, hop, nfft, zphase=True):
    """Inverse Short-time Fourier Transform.

    Perform iSTFT by overlap-add. See `ola` for meanings of arguments for
    synthesis.
    """
    def idft(frame):
        frame = irfft(frame)
        # from: [... x[-2] x[-1] 0 ... 0 x[0] x[1] ...]
        # to:   [0 ... x[0] x[1] ... x[-1] 0 ...]
        if zphase:
            frame = np.roll(frame, nfft//2)
        return frame
    return ola((idft(frame) for frame in sframes), sr, wind, hop)


def stlogm(sig, sr, wind, hop, nfft, trange=(0, None), ninf=-10.):
    """Short-time Log Magnitude Spectrum.

    Implement short-time log magnitude spectrum. Discrete frequency bins that
    have 0 magnitude are rounded to `floor` log magnitude. See `stana` for
    meanings of short-time analysis arguments.

    Arguments
    ---------
    nfft: int
        DFT size.

    Keyword Arguments
    -----------------
    trange: tuple [(0, None)]
        time range to be analyzed in seconds.
    ninf: float [-10.]
        negative infinities due to log(0) will be floored to the finite minimum
        magnitude in the current frame + ninf.
    Outputs
    -------
    sout: iterable of 1-D ndarray
        iterable of short-time log amplitdue spectra.
    """
    for frame in stana(sig, sr, wind, hop, trange):
        yield logmag(rfft(frame, n=nfft), ninf=ninf)


def strcep(sig, sr, wind, hop, n, trange=(0, None), ninf=-10.):
    """Short-time real cepstrum.

    Implement short-time (real) cepstrum. Discrete frequency bins that
    have 0 magnitude are rounded to `floor` log magnitude. See `stana` for
    meanings of short-time analysis arguments.

    Arguments
    ---------
    nfft: int
        DFT size.

    Keyword Arguments
    -----------------
    trange: tuple [(0, None)]
        time range to be analyzed in seconds.
    ninf: float [-10.]
        flooring for log(0)
    Outputs
    -------
    sout: iterable of 1-D ndarray
        iterable of short-time (real) cepstra. Note that each frame will now
        have length `nfft` instead of `len(wind)`. Having large `nfft` is good
        for dealing with time-aliasing in the quefrency domain.
    """
    for frame in stana(sig, sr, wind, hop, trange=trange):
        yield realcep(frame, n, ninf=ninf)


def stccep(sig, sr, wind, hop, n, trange=(0, None), ninf=-10.):
    """Short-time complex cepstrum."""
    for frame in stana(sig, sr, wind, hop, trange=trange):
        yield compcep(frame, n, ninf=ninf)


def stpsd(sig, sr, wind, hop, nfft, trange=(0, None), nframes=-1):
    """Estimate PSD by taking average of frames of PSDs (the Welch method).

    See `stana` and `stft` for detailed explanation of arguments.
    Arguments
    ---------
    sig: 1-D ndarray
        signal to be analyzed.
    Keyword Arguments
    -----------------
    nframes: int [None]
        maximum number of frames to be averaged. Default exhausts all frames.
    """
    psd = np.zeros(nfft//2+1)
    for ii, nframe in enumerate(stft(sig, sr, wind, hop, nfft, trange=trange)):
        psd += np.abs(nframe)**2  # collect PSD of all frames
        if (nframes > 0) and (ii+1 == nframes):
            break  # only average 6 frames
    psd /= (ii+1)  # average over all frames
    return psd


# Frame-level frequency-domain processing code below


def realcep(frame, n, nfft=4096, ninf=-10., comp=False, ztrans=False):
    """Compute real cepstrum of short-time signal `frame`.

    There are two modes for calculation:
        1. complex = False (default). This calculates c[n] using inverse DFT
        of the log magnitude spectrum.
        2. complex = True. This first calculates the complex cepstrum through
        the z-transform method (see `compcep`), and takes the even function to
        obtain c[n].
    In both cases, `len(cep) = nfft//2+1`.

    Arguments
    ---------
    frame: 1-D ndarray
        signal to be processed.
    nfft: non-negative int
        nfft//2+1 cepstrum in range [0, nfft//2] will be evaluated.
    Keyword Arguments
    -----------------
    ninf: float [-10.]
        flooring for log(0). Ignored if complex=True.
    complex: boolean [False]
        Use mode 2 for calculation.
    Outputs
    -------
    cep: 1-D ndarray
        Real ceptra of signal `frame` of:
        1. length `len(cep) = nfft//2+1`.
        2. quefrency index [0, nfft//2].
    """
    if comp:  # do complex method
        ccep = compcep(frame, n-1, ztrans=ztrans)
        rcep = .5*(ccep+ccep[::-1])
        return rcep[n-1:]  # only keep non-negative quefrency
    else:  # DFT method
        rcep = irfft(logmag(rfft(frame, nfft), ninf=ninf))
        return rcep[:n]


def compcep(frame, n, nfft=4096, ninf=-10., ztrans=False):
    """Compute complex cepstrum of short-time signal using Z-transform.

    Compute the aliasing-free complex cepstrum using Z-transform and polynomial
    root finder. Implementation is based on RS eq 8.68 on page 436.

    Arguments
    ---------
    frame: 1-D ndarray
        signal to be processed.
    n: non-negative int
        index range [-n, n] in which complex cepstrum will be evaluated.

    Outputs
    -------
    cep: 1-D ndarray
        complex ceptrum of length `2n+1`; quefrency index [-n, n].
    """
    if ztrans:
        frame = np.trim_zeros(frame)
        f0 = frame[0]
        roots = np.roots(frame/f0)
        rmag = np.abs(roots)
        assert 1 not in rmag
        ra, rb = roots[rmag < 1], roots[rmag > 1]
        amp = f0 * np.prod(rb)
        if len(rb) % 2:  # odd number of zeros outside UC
            amp = -amp
        # obtain complex cepstrum through eq (8.68) in RS, pp. 436
        cep = np.zeros(2*n+1)
        if rb.size > 0:
            for ii in range(-n, 0):
                cep[-n+ii] = np.real(np.sum(rb**ii))/ii
        cep[n] = np.log(np.abs(amp))
        if ra.size > 0:
            for ii in range(1, n+1):
                cep[n+ii] = -np.real(np.sum(ra**ii))/ii
    else:
        assert n <= nfft//2
        spec = rfft(frame, n=nfft)
        lmag, phase = lmagphase(spec, unwrap=True, ninf=ninf)
        cep = irfft(lmag+1j*phase, n=nfft)[:2*n+1]
        cep = np.roll(cep, n)
    return cep


def magphase(cspectrum, unwrap=False):
    """Return (magnitude, phase) of complex spectrum."""
    mag = np.abs(cspectrum)
    phs = np.angle(cspectrum)
    if unwrap:
        phs = np.unwrap(phs)
    return mag, phs


def lmagphase(cspectrum, unwrap=False, ninf=-10.):
    """Return (log-magnitude, phase) of complex spectrum."""
    mag, phs = magphase(cspectrum, unwrap=unwrap)
    lmag = logmag(cspectrum, ninf=ninf)
    return lmag, phs


def logmag(cspectrum, ninf=-10.):
    """Compute log magnitude of complex spectrum. Floor to `ninf` for -inf."""
    logspec = np.log(np.abs(cspectrum))
    if logspec.min() == -np.inf:
        if np.all(np.isinf(logspec)):
            return np.zeros_like(cspectrum)+(-50+ninf)
        fmin = np.min(logspec[logspec != -np.inf])
        logspec[logspec == -np.inf] = fmin + ninf
    return logspec


def phasor(mag, phase):
    """Compute complex spectrum given magnitude and phase."""
    return mag * np.exp(1j*phase)


def __frate2hsize__(sr, frate):
    """Translate frame rate in Hz to hop size in integer."""
    return int(sr*1.0/frate)


# Frame-level time-domain processing below

def lpc(frame, order, method='autocorr', levinson=False, out='full',
        force_stable=True):
    """Linear predictive coding (LPC).

    Arguments
    ---------
    frame: ndarray
        (Usually windowed) time-domain sequence
    order: int
        LPC order

    Keyword Arguments
    -----------------
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

    Outputs
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


def xcorr(x, y=None, one_side=True):
    r"""Calculate the cross-correlation between x and y.

    The cross-correlation is defined as:
        \phi_xy[k] = \sum_m x[m]*y[m+k]

    Arguments
    ---------
    x: ndarray
        A time sequence
    y: ndarray [None]
        Another time sequence; default to x if None.

    Keyword Arguments
    -----------------
    one_side: bool
        Returns one-sided correlation sequence starting at index 0 if
        True, otherwise returns the full sequence. This is only useful
        in the case where y is None.
    Outputs
    -------
    The cross-correlation sequence
    """
    if y is None:  # auto-correlation mode
        if one_side:  # return only one side of the symmetric sequence
            return fftconvolve(x[::-1], x)[len(x)-1:]
        else:  # return the entire symmetric sequence
            return fftconvolve(x[::-1], x)
    else:  # cross-correlation mode
        return fftconvolve(x[::-1], y)


def lpc2ref(alpha):
    """Convert a set of LPC alphas to reflection coefficients.

    Arguments
    ---------
    alpha: ndarray
        LPC coefficients (excluding 1)

    Outputs
    -------
    k: ndarray
        Reflection coefficients of the same order as alpha.
    """
    order = len(alpha)
    a = np.zeros((order, order))
    a[-1] = alpha
    for i in range(order-2, -1, -1):
        a[i, :i+1] = (a[i+1, :i+1]+a[i+1, i+1] *
                      np.flipud(a[i+1, :i+1]))/(1-a[i+1, i+1]**2)
    return np.diag(a)


def ref2lpc(k):
    """Convert a set of reflection coefficients `k` to LPC `alpha`.

    Arguments
    ---------
    k: ndarray
        reflection coefficients

    Outputs
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

    Arguments
    ---------
    alpha: ndarray
        LPC coefficients

    Outputs
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
    """Checl if the set of LPC coefficients is stable."""
    return ref_is_stable(lpc2ref(alpha))


def ref2lar(k):
    """Convert a set of reflection coefficients to log area ratio.

    Arguments
    ---------
    k: ndarray
        reflection coefficients
    Outputs
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
