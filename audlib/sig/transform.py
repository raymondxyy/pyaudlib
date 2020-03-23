"""Interface for common TRANSFORM functions.

This class provides easy-to-use interface for common transform functions,
using `stproc` for short-time processing, and `fbanks` for filterbank
analysis.

See Also
--------
stproc, fbanks

"""

import numpy as np

from numpy.fft import rfft, irfft
from scipy.fftpack import dct

from .stproc import numframes, stana, ola
from .temporal import xcorr
from .cepstral import rcep_dft, ccep_zt


def stft(sig, wind, hop, nfft, center=True, synth=False, zphase=False):
    """Short-time Fourier Transform.

    Implement STFT with the Fourier transform view. See `stana` for meanings
    of short-time analysis Parameters.

    Parameters
    ----------
    sig: array_like
        Signal as an ndarray.
    wind: array_like
        Window function as an ndarray.
    hop: float or int
        Window hop fraction in float or hop size in int.
    nfft: int
        DFT size.
    trange: tuple of float
        Starting and ending point in seconds.
    zphase: bool
        Do zero-phase STFT? Default to yes.

    Returns
    -------
    sout: ndarray with size (numframes, nfft//2+1)
        The complex STFT matrix.

    See Also
    --------
    stproc.stana

    """
    frames = stana(sig, wind, hop, synth=synth, center=center)
    if zphase:
        fsize = len(wind)
        woff = (fsize-(fsize % 2)) // 2
        zp = np.zeros((frames.shape[0], nfft-fsize))  # zero padding
        return rfft(np.hstack((frames[:, woff:], zp, frames[:, :woff])))
    else:
        return rfft(frames, n=nfft)


def istft(sframes, wind, hop, nfft, zphase=False):
    """Inverse Short-time Fourier Transform.

    Perform iSTFT by overlap-add. See `ola` for meanings of Parameters for
    synthesis.

    Parameters
    ----------
    sframes: array_like
        Complex STFT matrix.
    wind: 1-D ndarray
        Window function.
    hop: int, float
        Hop size or hop fraction of window.
    nfft: int
        Number of DFT points.
    zphase: bool
        Zero-phase STFT? Default to True.

    Returns
    -------
    sout: ndarray
        Reconstructed time series.

    See Also
    --------
    stproc.ola

    """
    frames = irfft(sframes, n=nfft)
    if zphase:
        # from: [... x[-2] x[-1] 0 ... 0 x[0] x[1] ...]
        # to:   [x[0] x[1] ... x[-1] 0 ...]
        fsize = len(wind)
        woff = (fsize-(fsize % 2)) // 2
        frames = np.concatenate((frames[:, (nfft-woff):],
                                 frames[:, :(fsize-woff)]), axis=1)
    else:
        frames = frames[:, :len(wind)]
    return ola(frames, wind, hop)


def stacf(sig, wind, hop, norm=True, biased=True):
    """Short-time autocorrelation function."""
    frames = stana(sig, wind, hop)
    return np.asarray([xcorr(f, norm=norm, biased=biased) for f in frames])


def stpowspec(sig, wind, hop, nfft, synth=False):
    """Short-time power spectrogram."""
    spec = stft(sig, wind, hop, nfft, synth=synth, zphase=False)
    return spec.real**2 + spec.imag**2


def stmelspec(sig, wind, hop, nfft, melbank, synth=False):
    """Short-time Mel frequency spectrogram."""
    return stpowspec(sig, wind, hop, nfft, synth=synth) @ melbank.wgts


def stmfcc(sig, wind, hop, nfft, melbank, synth=False):
    """Short-time Mel frequency ceptrum coefficients."""
    return dct(
        np.log(stmelspec(sig, wind, hop, nfft, melbank, synth=synth)),
        norm='ortho')


def strcep(sig, wind, hop, n, synth=False, nfft=4096, floor=-80.):
    """Short-time real cepstrum.

    Implement short-time (real) cepstrum. Discrete frequency bins that
    have 0 magnitude are rounded to `floor` log magnitude. See `stana` for
    meanings of short-time analysis Parameters.

    Parameters
    ---------
    nfft: int
        DFT size.
    synth: bool
        Aligned time frames with STFT synthesis.
    floor: float, -80
        Log-magnitude floor in dB.

    Returns
    -------
    sout: iterable of 1-D ndarray
        iterable of short-time (real) cepstra. Note that each frame will now
        have length `nfft` instead of `len(wind)`. Having large `nfft` is good
        for dealing with time-aliasing in the quefrency domain.

    """
    nframe = numframes(sig, wind, hop, synth=synth)
    out = np.empty((nframe, n))
    for ii, frame in enumerate(stana(sig, wind, hop, synth=synth)):
        out[ii] = rcep_dft(frame, n, nfft, floor)

    return out


def stccep(sig, wind, hop, n, synth=False):
    """Short-time complex cepstrum."""
    nframe = numframes(sig, wind, hop, synth=synth)
    out = np.empty((nframe, 2*n-1))
    for ii, frame in enumerate(stana(sig, wind, hop, synth=synth)):
        out[ii] = ccep_zt(frame, n)

    return out


def stpsd(sig, wind, hop, nfft, nframes=0):
    """Estimate PSD by taking average of frames of PSDs (the Welch method).

    See `stana` and `stft` for detailed explanation of Parameters.

    Parameters
    ----------
    sig: 1-D ndarray
        signal to be analyzed.
    nframes: int, 0
        Average the first n frames. Default (0) takes all frames.

    """
    spec = stft(sig, wind, hop, nfft, synth=False)
    if nframes == 0:
        return (spec.real**2 + spec.imag**2).mean(axis=0)

    assert nframes > 0, "Invalid argument"
    return (spec[:nframes].real**2 + spec[:nframes].imag**2).mean(axis=0)


def stcqt(sig, fr, cqbank):
    """Implement Judith Brown's Constant Q transform.

    Parameters
    ----------
    sig: array_like
        Signal to be processed.
    fr: int
        Frame rate in Hz, or int(SR/hop_length).
    cqbank: ConstantQ Filterbank class
        A pre-defined constant Q filterbank class.

    See Also
    --------
    fbank.ConstantQ

    """
    return cqbank.cqt(sig, fr)
