"""Utility functions related to audio processing."""

import numpy as np
from numpy.fft import fft
import numpy.random as rand
from scipy.signal import lfilter


def pre_emphasis(sig, alpha):
    """First-order highpass filter signal."""
    return lfilter([1, -alpha], 1, sig)


def dither(sig, scale):
    """Dither signal by adding small amount of noise to signal."""
    return sig + np.random.randn(*sig.shape)*scale


def firfreqz(h, nfft):
    """Compute frequency response of an FIR filter."""
    ww = np.linspace(0, 2, num=nfft, endpoint=False)
    hh = fft(h, n=nfft)
    return ww, hh


def nextpow2(n):
    """Give next power of 2 bigger than n."""
    return 1 << (n-1).bit_length()


def sample(x, length, num, verbose=False):
    """Given audio x, sample `num` segments with `length` samples each."""
    assert len(x) >= length
    segs = []
    start_idx_max = len(x)-length
    start_idx = np.around(rand.rand(num) * start_idx_max)
    for i in start_idx:
        segs.append(x[int(i):int(i)+length])
        if verbose:
            print('Take samples {} to {}...'.format(str(i), str(i+length)))
    return segs


def sample_pair(x, y, length, num, verbose=False):
    """Sample a pair of signals."""
    maxlength = min(len(x), len(y))
    assert maxlength >= length
    xsegs, ysegs = [], []
    start_idx_max = maxlength-length
    start_idx = np.around(rand.rand(num) * start_idx_max)
    for i in start_idx:
        xsegs.append(x[int(i):int(i)+length])
        ysegs.append(y[int(i):int(i)+length])
        if verbose:
            print('Take samples {} to {}...'.format(str(i), str(i+length)))
    return xsegs, ysegs


def add_noise(x, n, snr=None):
    """Add user provided noise n with SNR=snr to signal x.

    SNR = 10log10(Signal Energy/Noise Energy)
    NE = SE/10**(SNR/10)
    """
    # Take care of size difference in case x and n have different shapes
    xlen, nlen = len(x), len(n)
    if xlen > nlen:  # need to append noise several times to cover x range
        nn = np.tile(n, int(np.ceil(xlen/nlen)))
        nlen = len(nn)
    else:
        nn = n
    if xlen < nlen:  # slice a portion of noise
        nn = sample(nn, xlen, 1)[0]
    else:  # equal length
        nn = nn

    if snr is None:
        snr = (rand.random()-0.25)*20
    xe = x.dot(x)  # signal energy
    ne = nn.dot(nn)  # noise power
    nscale = np.sqrt(xe/(10**(snr/10.)) / ne)  # scaling factor
    return x + nscale*nn


def add_white_noise(x, snr=None):
    """Add white noise with SNR=snr to signal x.

    SNR = 10log10(Signal Energy/Noise Energy) = 10log10(SE/var(noise))
    var(noise) = SE/10**(SNR/10)
    """
    n = rand.normal(0, 1, x.shape)
    return add_noise(x, n, snr)


def white_noise(x, snr=None):
    """Return the white noise array given signal and desired SNR."""
    n = rand.normal(0, 1, x.shape)
    if snr is None:
        snr = (rand.random()-0.25)*20
    xe = x.dot(x)  # signal energy
    ne = n.dot(n)  # noise power
    nscale = np.sqrt(xe/(10**(snr/10.)) / ne)  # scaling factor

    return nscale*n


def normalize(x):
    """Normalize signal amplitude to be in range [-1,1]."""
    return x/np.max(np.abs(x))


def add_white_noise_rand(x):
    """Add white noise with SNR in range [-10dB,10dB]."""
    return add_white_noise(x, (rand.random()-0.25)*20)


def quantize(x, n):
    """Apply n-bit quantization to signal."""
    x /= np.ma.max(np.abs(x))  # make sure x in [-1,1]
    bins = np.linspace(-1, 1, 2**n+1, endpoint=True)  # [-1,1]
    qvals = (bins[:-1] + bins[1:]) / 2
    bins[-1] = 1.01  # Include 1 in case of clipping
    return qvals[np.digitize(x, bins)-1]
