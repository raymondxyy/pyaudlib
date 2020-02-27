"""Short-Time PROCessing of audio signals.

This module implements common audio analysis and synthesis techniques based on
short-time signal analysis.

See Also
--------
fbanks: Filterbank analysis and synthesis

"""

import math
from types import GeneratorType

import numpy as np
from numpy.lib.stride_tricks import as_strided

from .window import hop2hsize


def stcenters(sig, wind, hop, synth=False, center=False):
    """Calculate the window center in samples for each short-time frame.

    See `numframes` for meaning of the parameters.

    """
    ssize = len(sig)
    fsize = len(wind)
    hsize = hop2hsize(wind, hop)
    if synth:
        sstart = hsize-fsize  # int(-fsize * (1-hfrac))
    elif center:
        sstart = -int(len(wind)/2)  # odd window centered at exactly n=0
    else:
        sstart = 0
    send = ssize

    return (np.arange(sstart, send, hsize) + (fsize-1)/2.)


def numframes(sig, wind, hop, synth=False, center=False):
    """Calculate total number of short-time frames.

    Use this function to pre-determine the size of short-time transforms.

    Parameters
    ----------
    sig: array_like
        Signal to be analyzed.
    wind: array_like
        Window function.
    hop: float or int
        Hop fraction in (0, 1) or hop size in integers.
    synth: bool, False
        Whether short-time synthesis will be eventually used.
        This option has higher priority than center
    center: bool, False
        Whether the first frame is centered around 0.
        This option is ignored if synth=True.

    Returns
    -------
    out: int
        Number of frames to be computed by `stana`.

    See Also
    --------
    `stana`.

    """
    ssize = len(sig)
    fsize = len(wind)
    hsize = hop2hsize(wind, hop)
    if synth:
        sstart = hsize-fsize  # int(-fsize * (1-hfrac))
    elif center:
        sstart = -int(len(wind)/2)  # odd window centered at exactly n=0
    else:
        sstart = 0
    send = ssize

    return math.ceil((send-sstart)/hsize)


def stana(sig, wind, hop, synth=False, center=False):
    """[S]hort-[t]ime [Ana]lysis of audio signal by windowing.

    Parameters
    ----------
    sig: array_like
        Time series to be analyzed.
    wind: array_like
        Window function used for framing. See `window` for window functions.
    hop: float or int
        Hop fraction in (0, 1) or hop size in integers.
    synth: bool, False
        Whether time-domain synthesis is the end goal.
        This option has higher priority than `center` if enabled.
    center: bool, False
        Shift the windowed signal by half a window length if true. This is
        useful for applications like speech activity detection.
        When enabled, this option is only applied when `synth` is False.

    Returns
    -------
    frames: numpy.ndarray
        Short-time signal after windowing.

    See Also
    --------
    window.hamming: used to construct a valid window for analysis(/synthesis).

    """
    ssize = len(sig)
    fsize = len(wind)
    hsize = hop2hsize(wind, hop)
    if synth:
        sstart = hsize-fsize  # int(-fsize * (1-hfrac))
    elif center:
        sstart = -int(len(wind)/2)  # odd window centered at exactly n=0
    else:
        sstart = 0
    send = ssize

    nframe = math.ceil((send-sstart)/hsize)
    # Calculate zero-padding sizes
    zpleft = -sstart
    zpright = (nframe-1)*hsize+fsize - zpleft - ssize
    if zpleft > 0 or zpright > 0:
        sigpad = np.empty(ssize+zpleft+zpright, dtype=sig.dtype)
        sigpad[:zpleft] = 0
        sigpad[zpleft:len(sigpad)-zpright] = sig
        sigpad[len(sigpad)-zpright:] = 0
    else:
        sigpad = sig

    std = sigpad.strides[0]
    return as_strided(sigpad, shape=(nframe, fsize),
                      strides=(std*hsize, std)) * wind

    # Below is equivalent and more readable code for reference
    """
    frames = np.empty((math.ceil((send-sstart)/hsize), fsize))

    for ii, si in enumerate(range(sstart, send, hsize)):
        sj = si + fsize

        if si < 0:  # [0 0 ... x[0] x[1] ...]
            frames[ii] = np.pad(sig[:sj], (fsize-sj, 0), 'constant')
        elif sj > ssize:  # [... x[-2] x[-1] 0 0 ... 0]
            frames[ii] = np.pad(sig[si:], (0, fsize-ssize+si), 'constant')
        else:  # [x[..] ..... x[..]]
            frames[ii] = sig[si:sj]

    return frames * wind
    """


def ola(sframes, wind, hop):
    """Short-time Synthesis by [O]ver[l]ap-[A]dd.

    Perform the Overlap-Add algorithm on an array of short-time analyzed
    frames. Arguments used to call `stana` should be used here for consistency.
    Assume stana is called with trange set to default (i.e., permits perfect
    reconstruction).

    Parameters
    ----------
    sframes: array_like or iterable
        Array of short-time frames.
    wind: 1-D ndarray
        Window function.
    hop: int, float
        Hop size or hop fraction of window.

    Returns
    -------
    sout: ndarray
        Reconstructed time series.

    See Also
    --------
    stana

    """
    if type(sframes) is GeneratorType:
        sframes = np.asarray(list(sframes))

    nframe = len(sframes)
    fsize = len(wind)
    hsize = hop2hsize(wind, hop)
    hfrac = hsize*1. / fsize
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


def frate2hsize(sr, frate):
    """Translate frame rate in Hz to hop size in integer."""
    return int(sr*1.0/frate)
