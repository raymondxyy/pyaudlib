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
from .spectral import logmag, realcep, compcep
from .temporal import xcorr
from .auditory import hz2mel, dft2mel


def stft(sig, sr, wind, hop, nfft, center=True, synth=False, zphase=False):
    """Short-time Fourier Transform.

    Implement STFT with the Fourier transform view. See `stana` for meanings
    of short-time analysis Parameters.

    Parameters
    ----------
    sig: array_like
        Signal as an ndarray.
    sr: int
        Sampling rate.
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
    frames = stana(sig, sr, wind, hop, synth=synth, center=center)
    if zphase:
        fsize = len(wind)
        woff = (fsize-(fsize % 2)) // 2
        zp = np.zeros((frames.shape[0], nfft-fsize))  # zero padding
        return rfft(np.hstack((frames[:, woff:], zp, frames[:, :woff])))
    else:
        return rfft(frames, n=nfft)


def istft(sframes, sr, wind, hop, nfft, zphase=False):
    """Inverse Short-time Fourier Transform.

    Perform iSTFT by overlap-add. See `ola` for meanings of Parameters for
    synthesis.

    Parameters
    ----------
    sframes: array_like
        Complex STFT matrix.
    sr: int
        Sampling rate.
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
    return ola(frames, sr, wind, hop)


def stacf(sig, sr, wind, hop, norm=True, biased=True):
    """Short-time autocorrelation function."""
    frames = stana(sig, sr, wind, hop)
    return np.asarray([xcorr(f, norm=norm, biased=biased) for f in frames])


def stpowspec(sig, sr, wind, hop, nfft, synth=False):
    """Short-time power spectrogram."""
    spec = stft(sig, sr, wind, hop, nfft, synth=synth, zphase=False)
    return spec.real**2 + spec.imag**2


def stmelspec(sig, sr, wind, hop, nfft, melbank, synth=False):
    """Short-time Mel frequency spectrogram."""
    return stpowspec(sig, sr, wind, hop, nfft, synth=synth) @ melbank.wgts


def stmfcc(sig, sr, wind, hop, nfft, melbank, synth=False):
    """Short-time Mel frequency ceptrum coefficients."""
    return dct(
        np.log(stmelspec(sig, sr, wind, hop, nfft, melbank, synth=synth)),
        norm='ortho')


def stlogm(sig, sr, wind, hop, nfft, synth=False, floor=-10.):
    """Short-time Log Magnitude Spectrum.

    Implement short-time log magnitude spectrum. Discrete frequency bins that
    have 0 magnitude are rounded to `floor` log magnitude. See `stft` for
    complete documentation for each parameter.

    See Also
    --------
    stft

    """
    return logmag(stft(sig, sr, wind, hop, nfft, synth=synth, zphase=False),
                  floor=floor)


def strcep(sig, sr, wind, hop, n, synth=False, floor=-10.):
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
    floor: float [-10.]
        flooring for log(0)

    Returns
    -------
    sout: iterable of 1-D ndarray
        iterable of short-time (real) cepstra. Note that each frame will now
        have length `nfft` instead of `len(wind)`. Having large `nfft` is good
        for dealing with time-aliasing in the quefrency domain.

    """
    nframe = numframes(sig, sr, wind, hop, synth=synth)
    out = np.empty((nframe, n))
    for ii, frame in enumerate(stana(sig, sr, wind, hop, synth=synth)):
        out[ii] = realcep(frame, n, floor=floor)

    return out


def stccep(sig, sr, wind, hop, n, synth=False, floor=-10.):
    """Short-time complex cepstrum."""
    nframe = numframes(sig, sr, wind, hop, synth=synth)
    out = np.empty((nframe, 2*n+1))
    for ii, frame in enumerate(stana(sig, sr, wind, hop, synth=synth)):
        out[ii] = compcep(frame, n, floor=floor)

    return out


def stpsd(sig, sr, wind, hop, nfft, nframes=-1):
    """Estimate PSD by taking average of frames of PSDs (the Welch method).

    See `stana` and `stft` for detailed explanation of Parameters.

    Parameters
    ----------
    sig: 1-D ndarray
        signal to be analyzed.
    nframes: int [None]
        maximum number of frames to be averaged. Default exhausts all frames.

    """
    psd = np.zeros(nfft//2+1)
    for ii, nframe in enumerate(stft(sig, sr, wind, hop, nfft, synth=False)):
        psd += np.abs(nframe)**2  # collect PSD of all frames
        if (nframes > 0) and (ii+1 == nframes):
            break  # only average 6 frames
    psd /= (ii+1)  # average over all frames
    return psd


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

# Legacy functions below


def audspec(pspectrum, nfft=512, sr=16000., nfilts=0, fbtype='mel',
            minfrq=0, maxfrq=8000., sumpower=True, bwidth=1.0):
    if nfilts == 0:
        nfilts = np.int(np.ceil(hz2mel(np.array([maxfrq]), sphinx=False)[0]/2))
    if fbtype == 'mel':
        wts, _ = dft2mel(nfft, sr=sr, nfilts=nfilts, width=bwidth,
                         minfrq=minfrq, maxfrq=maxfrq)
    else:
        raise ValueError('Filterbank type not supported.')

    nframes, nfrqs = pspectrum.shape
    wts = wts[:, :nfrqs]

    if sumpower:  # weight power
        aspectrum = pspectrum.dot(wts.T)
    else:  # weight magnitude
        aspectrum = (np.sqrt(pspectrum).dot(wts.T))**2

    return aspectrum, wts


def invaudspec(aspectrum, nfft=512, sr=16000., nfilts=0, fbtype='mel',
               minfrq=0., maxfrq=8000., sumpower=True, bwidth=1.0):
    # TODO: Either update or remove this.
    if fbtype == 'mel':
        wts, _ = dft2mel(nfft, sr=sr, nfilts=nfilts, width=bwidth,
                         minfrq=minfrq, maxfrq=maxfrq)
    else:
        raise ValueError('Filterbank type not supported.')

    nframes, nfilts = aspectrum.shape
    # Cut off 2nd half
    wts = wts[:, :((nfft/2)+1)]

    # Just transpose, fix up
    ww = wts.T.dot(wts)
    iwts = wts / np.matlib.repmat(np.maximum(np.mean(np.diag(ww))/100.,
                                             np.sum(ww, axis=0)), nfilts, 1)

    #iwts = np.linalg.pinv(wts).T
    # Apply weights
    if sumpower:  # weight power
        spec = aspectrum.dot(iwts)
    else:  # weight magnitude
        spec = (np.sqrt(aspectrum).dot(iwts))**2
    return spec


def invaudspec_mask(aspectrum, weights):
    # TODO: Either update or remove this.
    energy = np.ones_like(aspectrum).dot(weights)
    mask = aspectrum.dot(weights)
    not_zero_energy = ~(energy == 0)
    mask[not_zero_energy] /= energy[not_zero_energy]
    return mask
