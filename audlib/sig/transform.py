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

from .stproc import numframes, stana, ola
from .spectral import logmag, realcep, compcep
from .auditory import hz2mel, dft2mel


# Short-time transforms

def stft(sig, sr, wind, hop, nfft, trange=(None, None), zphase=True):
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
    fsize = len(wind)
    woff = (fsize-(fsize % 2)) // 2
    zp = np.zeros(nfft-fsize)  # zero padding
    nframe = numframes(sig, sr, wind, hop, trange=trange)
    out = np.empty((nframe, nfft//2+1), dtype=np.complex_)
    for ii, frame in enumerate(stana(sig, sr, wind, hop, trange=trange)):
        if zphase:
            out[ii] = rfft(np.concatenate((frame[woff:], zp, frame[:woff])))
        else:  # conventional linear-phase STFT
            out[ii] = rfft(np.concatenate((frame, zp)))

    return out


def istft(sframes, sr, wind, hop, nfft, zphase=True):
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
    def idft(frame):
        frame = irfft(frame)
        # from: [... x[-2] x[-1] 0 ... 0 x[0] x[1] ...]
        # to:   [0 ... x[0] x[1] ... x[-1] 0 ...]
        if zphase:
            frame = np.roll(frame, nfft//2)
        return frame
    return ola(np.asarray([idft(frame) for frame in sframes]), sr, wind, hop)


def stlogm(sig, sr, wind, hop, nfft, trange=(0, None), floor=-10.):
    """Short-time Log Magnitude Spectrum.

    Implement short-time log magnitude spectrum. Discrete frequency bins that
    have 0 magnitude are rounded to `floor` log magnitude. See `stft` for
    complete documentation for each parameter.

    See Also
    --------
    stft

    """
    return logmag(stft(sig, sr, wind, hop, nfft, trange=trange, zphase=False),
                  floor=floor)


def strcep(sig, sr, wind, hop, n, trange=(0, None), floor=-10.):
    """Short-time real cepstrum.

    Implement short-time (real) cepstrum. Discrete frequency bins that
    have 0 magnitude are rounded to `floor` log magnitude. See `stana` for
    meanings of short-time analysis Parameters.

    Parameters
    ---------
    nfft: int
        DFT size.
    trange: tuple [(0, None)]
        time range to be analyzed in seconds.
    floor: float [-10.]
        flooring for log(0)

    Returns
    -------
    sout: iterable of 1-D ndarray
        iterable of short-time (real) cepstra. Note that each frame will now
        have length `nfft` instead of `len(wind)`. Having large `nfft` is good
        for dealing with time-aliasing in the quefrency domain.

    """
    nframe = numframes(sig, sr, wind, hop, trange=trange)
    out = np.empty((nframe, n))
    for ii, frame in enumerate(stana(sig, sr, wind, hop, trange=trange)):
        out[ii] = realcep(frame, n, floor=floor)

    return out


def stccep(sig, sr, wind, hop, n, trange=(0, None), floor=-10.):
    """Short-time complex cepstrum."""
    nframe = numframes(sig, sr, wind, hop, trange=trange)
    out = np.empty((nframe, n))
    for ii, frame in enumerate(stana(sig, sr, wind, hop, trange=trange)):
        out[ii] = compcep(frame, n, floor=floor)

    return out


def stpsd(sig, sr, wind, hop, nfft, trange=(0, None), nframes=-1):
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
    for ii, nframe in enumerate(stft(sig, sr, wind, hop, nfft, trange=trange)):
        psd += np.abs(nframe)**2  # collect PSD of all frames
        if (nframes > 0) and (ii+1 == nframes):
            break  # only average 6 frames
    psd /= (ii+1)  # average over all frames
    return psd


def cqt(x, fs, fmin=20., fmax=None, bins_per_octave=12, decimate=0):
    """Implement Judy Brown's Constant Q transform.

    TODO
    """
    assert fmin > 0.

    if fmax is not None:
        if fmax > fs/2.:
            print("`fmax` goes beyond Nyquist rate! Set to Nyquist rate.")
            fmax = fs/2.
    else:
        fmax = fs/2.

    num_octaves = int(np.floor(np.log2(fmax/fmin)))
    exponents = np.linspace(0, num_octaves, num=bins_per_octave*num_octaves)

    # Calculate center frequencies
    fc = fmin * (2.**exponents)

    # Calculate quality factor
    Q = 1. / (2.**(1./bins_per_octave)-1)

    # Calculate the window length for each filterbank
    Nk = np.round((fs / fc) * Q)

    # Calculate decimation factor
    # maximum decimation factor for hamming window. Assuming BW=4pi/(N-1)
    # where N is the window length
    L = np.floor((Nk[-1]-1)/4)  # pick minimum window length
    if decimate > L:
        print("Recommended decimation factor is [{}] or below".format(L))
        #decimate = L

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
    energy = np.ones_like(aspectrum).dot(weights)
    mask = aspectrum.dot(weights)
    not_zero_energy = ~(energy == 0)
    mask[not_zero_energy] /= energy[not_zero_energy]
    return mask
