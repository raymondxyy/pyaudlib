# Time-Frequency Analysis of Audio Signals
# Author: Raymond Xia (yangyanx@andrew.cmu.edu)

import numpy as np
import scipy.signal as sig
from numpy.fft import fft, ifft, rfft, irfft
from pdb import set_trace
from .window import hamming
from scipy.signal import lfilter
# Macros
__window_types__ = ['hamming', 'rectangular']
TWOPI = 2*np.pi


def stft_full(x, fs, window_type='hamming', window_length=0.064, NFFT=1024,
              stft_view='fourier_transform', stft_truncate=False):
    """
    stft: Perform short-time Fourier transform on signal x WITHOUT
    downsampling in either time or frequency domain.
    Args:
        x   - signal as column vector
        fs  - sampling frequency
        wtype - window function as string. Possible windows are:
           * hamming
           * hanning
           * rectangular
        wlen - window length in seconds
        NFFT - frequency resolution on [0,2pi)
    Returns:
        X   - STFT of x; dimension [T*NFFT]
        time_map - index-time mapping
        freq_map - index-frequency mapping
    """
    assert window_type in __window_types__
    wlen = int(window_length * fs)  # length in samples
    assert NFFT >= wlen
    if window_type == 'hamming':  # no need to reverse
        w = hamming(wlen)
    elif window_type == 'hanning':  # no need to reverse
        w = np.hanning(wlen)
    elif window_type == 'rectangular':  # no need to reverse
        w = np.ones(wlen)
    N = NFFT  # frequency bins in [0,2pi)
    hlen = 1  # hop size for adjacent time frames

    w_start = -wlen+1  # starting time point for window
    w_end = len(x)  # ending (exclusive) time point for window
    T = int(np.ceil((w_end-w_start) / hlen))  # number of time frames

    # compute real time-frequency index
    time_map = np.arange(T*1.0) * hlen / fs
    freq_map = np.arange(N*1.0) / N * fs

    if stft_view == 'fourier_transform':
        X = np.zeros((T, N))
        for t, n in enumerate(np.arange(w_start, w_end, hlen)):
            ix1_x, ix2_x = max(n, 0), min(n+wlen, w_end)  # x: frame [ix1,ix2)
            ix1_X = ix1_x-n  # real index for X
            ix2_X = ix1_X+(ix2_x-ix1_x)
            X[t, ix1_X:ix2_X] = x[ix1_x:ix2_x]
            X[t, :wlen] *= w  # window the actual segment
        if not stft_truncate:  # for general complex sequence
            return fft(X), time_map, freq_map
        else:  # for real sequence
            return rfft(X), time_map, freq_map[:X.shape[1]/2+1]
    elif stft_view == 'filterbank':
        N_ret = N if not stft_truncate else N/2+1
        X = np.zeros((T, N_ret), dtype=np.complex_)
        for k in np.arange(N_ret):
            X[:, k] = sig.lfilter(
                w, 1, x*np.exp(-1j*TWOPI*k/N*np.arange(w_end)))
        if stft_truncate:
            freq_map = freq_map[:N_ret]
        return X, time_map, freq_map
    else:
        raise ValueError('View can only be: fourier_transform/filterbank')


def istft_full(X, fs, window_length=0.064, method='ola', stft_truncate=False):
    '''
    istft: Perform inverse short-time fourier transform on X
        X   - STFT matrix [T,N]
      wtype - window function as string. Possible windows are:
           * hamming
           * hanning
           * rectangular
      wlen  - window length in integer
        hop - hop size [0,1]
        x   = re-constructed time domain signal
    '''

    wlen = int(window_length * fs)  # length in samples
    w = hamming(wlen)

    if method == 'ola':  # synthesize using overlap add
        T, N = X.shape
        x_frames = irfft(X) if stft_truncate else ifft(X)
        x = np.zeros(T-1+wlen, dtype=np.complex_)
        for t in range(T):
            x[t:t+wlen] += x_frames[t, :wlen]
        x /= np.sum(w)
        return np.real(x[wlen-1:-(wlen-1)])  # crop to original length
    elif method == 'fbs':  # synthesize using filterbank synthesis
        if stft_truncate:  # fill up Hermeatian symmetric part
            end = -1 if wlen % 2 else -2  # last channel to be preserved
            X = np.concatenate((X, np.conj(X[:, end:0:-1])), axis=1)
        T, N = X.shape
        X *= np.exp(np.outer(np.arange(T), 1j*TWOPI*np.arange(N)/N))
        return np.real(np.sum(X, axis=1))[:-(wlen-1)] / (w[0]*N)
    else:
        raise ValueError('Synthesis method has to be ola/fbs.')


def stft(x, fs,
         w_start=None, w_end=None,
         window_type='hamming', window_length=0.032, hop_fraction=0.25,
         nfft=512, stft_truncate=True, tf_unit='continuous', stft_full=False):
    '''
    stft: Perform short-time Fourier transform on signal x
    This is the decimated version of the STFT for efficiency.
    NFFT = wlen because of frequency decimation.
    Args:
        x   - signal as column vector
        fs  - sampling frequency
        wtype - window function as string. Possible windows are:
           * hamming
           * hanning
           * rectangular
        wlen - window length in seconds
        hop_fraction - hop fraction [0,1]
        stft_truncate - True for real sequence to remove redundancy
    Returns:
        X   - STFT of x; dimension [T*N]
        time_map - index-time mapping
        freq_map - index-frequency mapping
    '''
    assert stft_full or ((hop_fraction > 0) and (hop_fraction < 1))
    assert window_type in __window_types__

    # construct window
    if window_length < 1:  # assume unit in seconds
        wlen = int(window_length * fs)  # length in samples
    else:  # assume unit in samples
        wlen = window_length
    woffset = int(wlen-(wlen % 2))//2  # the window symmetry point

    if window_type == 'hamming':
        w = hamming(wlen)
        w /= (0.54/hop_fraction)
        #assert hop_fraction <= .25 # avoid aliasing
    elif window_type == 'rectangular':
        w = np.ones(wlen)
        w /= (1/hop_fraction)
    else:
        raise ValueError('Unsupported window type!')

    # Pad input if shorter than window
    xlen = len(x)
    if xlen < wlen:
        print("Padding x in make it at least as long as window. Input changed.")
        x = np.pad(x, (0, wlen-xlen), 'constant')
        xlen = wlen

    if stft_full:  # inefficient. only here for demonstration purpose
        hlen = 1
        nfft = 1024
    else:
        hlen = int(np.floor(hop_fraction * wlen))

    # starting x[n] time point for window
    if w_start is None:
        w_start = -int(wlen*(1-hop_fraction))
    if w_end is None:
        w_end = xlen  # ending (exclusive) x[n] time point for window
    T = int(np.ceil((w_end-w_start)*1.0 / hlen))  # number of time frames

    # compute real (physical) time-frequency index
    # calculates the center of window in physical time unit for each frame
    if tf_unit == 'continuous':
        time_map = (np.arange(T*1.0)*hlen + w_start +
                    woffset) / fs  # unit in seconds
        freq_map = np.arange(nfft*1.0) / nfft * fs  # unit in herz
    elif tf_unit == 'discrete':  # otherwise digital
        time_map = (np.arange(T)*hlen + w_start+woffset)  # unit in samples
        freq_map = np.arange(nfft) / nfft  # unit in discrete frequency
    else:
        raise ValueError('Option not supported for `tf_unit`')

    # implement decimated STFT with FT view
    if stft_truncate:  # keep half of frequency bins for symmetric spectrum
        X = np.empty((T, nfft//2+1), dtype=np.complex)
    else:  # for general complex sequence
        X = np.empty((T, nfft), dtype=np.complex)
    #print X.shape
    zp = np.zeros(nfft-wlen)  # zero padding
    for t, n in enumerate(np.arange(w_start, w_end, hlen)):
        xframe = np.zeros(wlen)  # frame buffer
        if n < 0:  # starting frames
            xframe[-n:] = x[:n+wlen]  # [0 0 ... x[0] x[1] ...]
        elif n+wlen > xlen:  # ending frames
            xframe[:xlen-n] = x[n:]  # [... x[-2] x[-1] 0 0 ... 0]
        else:  # others
            xframe[:] = x[n:n+wlen]
        xframe *= w  # window the actual segment
        #if time_normalize: # circular shift to make phase response consistent
        #    xframe = np.roll(xframe,t*hlen)
        xframe_padded = np.concatenate(
            (xframe[woffset:], zp, xframe[:woffset]))
        if stft_truncate:
            X[t, :] = rfft(xframe_padded)
        else:
            X[t, :] = fft(xframe_padded)

    if stft_truncate:
        return time_map, freq_map[:X.shape[1]], X
    else:  # for real sequence
        return time_map, freq_map, X


def istft(X, fs, end_time, window_length=0.064, hop_fraction=0.25, end_sec=False,
          method='ola', stft_truncate=True):
    '''
    istft: Perform inverse short-time fourier transform on X
        X   - decimated STFT matrix [T,N]
      wtype - window function as string. Possible windows are:
           * hamming
           * hanning
           * rectangular
      wlen  - window length in integer
        hop - hop size [0,1]
        x   = re-constructed time domain signal
    '''
    if window_length < 1:  # unit in seconds
        wlen = int(window_length * fs)  # window length in samples
    else:  # unit in samples
        wlen = window_length
    #w = hamming(wlen)
    hlen = int(np.floor(hop_fraction * wlen))  # hop size in samples
    if end_sec:
        idx_end = int(end_time * fs)
    else:
        idx_end = end_time

    if method == 'ola':  # synthesize with overlap-add
        x_frames = irfft(X) if stft_truncate else ifft(X)
        T, nfft = x_frames.shape
        f_offset = nfft/2  # assume even nfft
        # from: [... x[-2] x[-1] 0 ... 0 x[0] x[1] ...]
        # to:   [0 ... x[0] x[1] ... x[-1] 0 ...]
        x_frames = np.roll(x_frames, f_offset, axis=1)
        if stft_truncate:
            x = np.zeros(idx_end)
        else:
            x = np.zeros(idx_end, dtype=np.complex)
        #xoff = -f_offset # starting time index for OLA
        xoff = -int(wlen*(1-hop_fraction))
        for t in range(T):
            # handle boundary case
            if (xoff+wlen <= 0):
                # Entire frame not hitting n=0; assuming all 0s
                xoff += hlen
                continue
            if (xoff >= idx_end):
                break

            if xoff < 0:  # first few frames not entering n=0
                x[:xoff+nfft] += x_frames[t, -xoff:]
            elif xoff+nfft > len(x):
                x[xoff:] += x_frames[t, :(len(x)-xoff)]
            else:
                x[xoff:xoff+nfft] += x_frames[t, :]
            xoff += hlen
    elif method == 'fbs':  # TODO
        raise ValueError('Not implemented yet.')
    else:
        raise ValueError('Synthesis method has to be ola/fbs')

    if stft_truncate:
        return x
    else:
        return np.real(x)


def cqt(x, fs, fmin=20., fmax=None, bins_per_octave=12, decimate=0):
    """
    Implement Judy Brown's Constant Q transform.
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


def magphase(spec):
    """
    Decompose a complex spectrogram into magnitude and phase
    X = M * P
    """
    mspec = np.abs(spec)
    pspec = np.empty_like(spec)
    zero_mag = (mspec == 0.)  # fix zero-magnitude
    pspec[zero_mag] = 1.
    pspec[~zero_mag] = spec[~zero_mag]/mspec[~zero_mag]
    return mspec, pspec


def logspec(X, floor=-15):
    """
    Take log of power spectrum X. Prevent log(0)
    """
    zero_mag = X == 0
    logX = np.zeros_like(X)
    logX[zero_mag] = floor
    logX[~zero_mag] = np.log10(X[~zero_mag])
    return logX


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


def hz2mel(f, sphinx):
    if sphinx:
        return 2595. * np.log10(1+f/700.)
    # match Slaney's toolbox
    f0, f_sp, brkfrq = 0., 200./3, 1000.
    brkpt = (brkfrq - f0) / f_sp
    logstep = np.exp(np.log(6.4)/27.)

    z = np.empty_like(f)
    lower = f < brkfrq  # np.less(f,brkfrq)
    higher = np.logical_not(lower)

    z[lower] = (f[lower] - f0) / f_sp
    z[higher] = brkpt + np.log(f[higher]/brkfrq) / np.log(logstep)
    return z


def mel2hz(z, sphinx):
    if sphinx:
        return 700*(10**(z/2595.)-1)

    f0, f_sp, brkfrq = 0., 200./3, 1000.
    brkpt = (brkfrq - f0) / f_sp
    logstep = np.exp(np.log(6.4)/27.)

    f = np.empty_like(z)
    lower = z < brkpt  # np.less(z,brkpt)
    higher = np.logical_not(lower)

    f[lower] = f0 + z[lower] * f_sp
    f[higher] = brkfrq * np.exp(np.log(logstep)*(z[higher]-brkpt))
    return f


def dft2mel(nfft, sr=8000., nfilts=0, width=1., minfrq=0., maxfrq=4000.,
            sphinx=False, constamp=True):
    '''
    dft2mel: Generate a weight matrix that maps linear discrete frequencies to
    Mel scale.
    '''

    if nfilts == 0:
        nfilts = np.int(np.ceil(hz2mel(np.array([maxfrq]), sphinx)[0]/2))

    weights = np.zeros((nfilts, nfft))

    # dft index -> linear frequency in hz
    dftfrqs = np.arange(nfft/2+1, dtype=np.float)/nfft * sr

    maxmel, minmel = hz2mel(np.array([maxfrq, minfrq]), sphinx)
    binfrqs = mel2hz(minmel+np.linspace(0., 1., nfilts+2)
                     * (maxmel-minmel), sphinx)
    #set_trace()

    for i in range(nfilts):
        fs = binfrqs[i:i+3].copy()
        fs = fs[1] + width*(fs-fs[1])  # adjust bandwidth if needed
        loslope = (dftfrqs - fs[0])/(fs[1] - fs[0])
        hislope = (fs[2] - dftfrqs)/(fs[2] - fs[1])
        weights[i, 0:nfft/2+1] = np.maximum(0, np.minimum(loslope, hislope))

    if constamp:
        # Slaney-style mel is scaled to be approx constant E per channel
        weights = np.diag(
            2/(binfrqs[2:nfilts+2]-binfrqs[:nfilts])).dot(weights)
    weights[:, nfft/2+1:] = 0  # avoid aliasing

    return weights, binfrqs[1:]


def dct1(x, dft=False):
    """
    Perform Type-1 Discrete Cosine Transform (DCT-1) on input signal `x`.
    Parameters
    ----------
    x: numpy array
        input signal
    dft: boolean
        implement using dft?

    Returns
    -------
    X: numpy array
        Type-1 DCT of x.
    """
    if len(x) == 1:
        return x.copy()
    ndct = len(x)

    if dft:  # implement using dft
        x_ext = np.concatenate((x, x[-2:0:-1]))  # create extended sequence
        X = np.real(rfft(x_ext)[:ndct])
    else:  # implement using definition
        xa = x * 1.
        xa[1:-1] *= 2.  # form x_a sequence
        X = np.zeros_like(xa)
        ns = np.arange(ndct)
        for k in range(ndct):
            cos = np.cos(np.pi*k*ns/(ndct-1))
            X[k] = cos.dot(xa)
    return X


def idct1(X, dft=False):
    """
    Perform inverse Type-1 Discrete Cosine Transform (iDCT-1) on spectrum `X`.
    Parameters
    ----------
    X: numpy array
        input DCT spectrum.
    dft: boolean
        implement using dft?

    Returns
    -------
    x: numpy array
        inverse Type-1 DCT of X.
    """
    if len(X) == 1:
        return X.copy()
    ndct = len(X)

    if dft:  # implement using dft
        x = irfft(X, n=2*(ndct-1))[:ndct]
    else:  # implement using definition
        Xb = X / (ndct-1.)
        Xb[0] /= 2.
        Xb[-1] /= 2.
        x = np.zeros_like(Xb)
        ks = np.arange(ndct)
        for n in range(ndct):
            cos = np.cos(np.pi*n*ks/(ndct-1))
            x[n] = cos.dot(Xb)
    return x


def dct2(x, normalize=True, dft=False):
    """
    Perform Type-2 Discrete Cosine Transform (DCT-2) on input signal `x`.
    Parameters
    ----------
    x: numpy array
        input signal
    normalize: boolean
        do normalization so that energy is preserved.
    dft: boolean
        implement using dft?

    Returns
    -------
    X: numpy array
        Type-2 DCT of x.
    """
    if len(x) == 1:
        return x.copy()
    ndct = len(x)

    if dft:  # implement using dft
        if normalize:
            raise ValueError("DFT method does not support normalization!")
        Xk = rfft(x, 2*ndct)[:ndct]
        X = 2*np.real(Xk*np.exp(-1j*(np.pi*np.arange(ndct)/(2*ndct))))
    else:  # implement using definition
        if normalize:
            xa = 1.*x
        else:
            xa = 2.*x
        X = np.zeros_like(xa)
        ns = np.arange(ndct)
        for k in range(ndct):
            cos = np.cos(np.pi*k*(2*ns+1)/(2*ndct))
            X[k] = cos.dot(xa)
            if normalize:
                X[k] *= np.sqrt(2./ndct)
        if normalize:
            X[0] /= np.sqrt(2)
    return X


def idct2(X, normalize=True, dft=False):
    """
    Perform inverse Type-2 Discrete Cosine Transform (DCT-2) on input spectrum `X`.
    Parameters
    ----------
    X: numpy array
        input signal
    normalize: boolean
        do normalization so that energy is preserved.
    dft: boolean
        implement using dft?

    Returns
    -------
    x: numpy array
        inverse Type-2 DCT of X.
    """
    if len(X) == 1:
        return X.copy()
    ndct = len(X)

    if dft:  # implement using dft
        if normalize:
            raise ValueError("DFT method does not support normalization!")
        ks = np.arange(ndct)
        Xseg1 = X*np.exp(1j*np.pi*ks/(2*ndct))
        Xseg2 = -X[-1:0:-1]*np.exp(1j*np.pi*(ks[1:]+ndct)/(2*ndct))
        X_ext = np.concatenate((Xseg1, [0.], Xseg2))
        x = irfft(X_ext[:ndct+1])[:ndct]
    else:  # implement using definition
        if normalize:
            Xb = X * np.sqrt(2./ndct)
            Xb[0] /= np.sqrt(2.)
        else:
            Xb = X / (ndct+0.0)
            Xb[0] /= 2.
        x = np.zeros_like(Xb)
        ks = np.arange(ndct)
        for n in range(ndct):
            cos = np.cos(np.pi*ks*(2*n+1)/(2*ndct))
            x[n] = cos.dot(Xb)
    return x


def pre_emphasis(x, alpha):
    """
    "Pre-emphasis" of speech by simple high-pass filter.
    """
    return lfilter([1, -alpha], 1, x)


def dither(x, scale):
    return x + np.random.randn(*x.shape)*scale


def mfcc(x, sample_rate, frame_rate, frame_length, nfft=512, alpha=.97,
         nfilts=40, minfrq=0, maxfrq=8000., sumpower=True, bwidth=1.0,
         sphinx=True, numcep=13, dith=True):
    pass
