"""Some algorithms for PHASE estimation."""
import numpy as np
from numpy.fft import irfft, rfft

from .transform import stft
from .stproc import ola, stana, hop2hsize
from .spectral import magphasor


def griffin_lim(magspec, wind, hop, nfft, niter=32, init=None):
    """The Griffin-Lim iterative phase reconstruction method.

    This implementation follows Mark Lindsey's MATLAB implementation.

    Parameters
    ----------
    magspec: numpy.ndarray
        A time-by-frequency real matrix describing the magnitude spectrogram.
    wind: numpy.ndarray
        A real array describing the window function.

    Keyword Parameters
    ------------------
    niter: int
        Number of iterative estimation to do.
    init: numpy.ndarray or float
        Normalized initial phase value in (x\pi) radians. Default to zero phase.

    Returns
    -------
    phasor, err

    """
    we = (wind**2).sum()
    assert we > 0, "Window must not be complete zeros!"
    assert niter > 0, "Invalid number of iterations."

    # Init phase spectra
    if init is None:
        phasor = np.exp(1j*np.pi*(np.random.random_sample(magspec.shape)*2-1))
    elif type(init) in [float, int]:
        phasor = np.full_like(magspec, np.exp(1j*np.pi*init), dtype='complex128')
    else:
        assert magspec.shape == init.shape, "Incompatible init phase dimension."
        phasor = np.exp(1j*np.pi*init)

    # Define inverse STFT with weighted signal
    def stft_(sig):
        frames = stana(sig, wind, hop)
        return rfft(frames, nfft)

    def istft_(spec):
        """NOTE: stproc.ola does not work for GL because of alignment."""
        frames = irfft(spec, n=len(wind)) * wind / we
        nframe = len(frames)
        fsize = len(wind)
        hsize = hop2hsize(wind, hop)
        ssize = hsize*(nframe-1)+fsize
        ii = 0

        sout = np.zeros(ssize)
        for frame in frames:
            frame = frame[:fsize]
            sout[ii:ii+fsize] += frame
            ii += hsize

        return sout

    err = []
    for _ in range(niter):
        sig = istft_(magspec * phasor)
        magspec_, phasor = magphasor(stft_(sig))
        err.append(((magspec-magspec_[:len(magspec)])**2).mean())
        phasor = phasor[:len(magspec)]

    return sig, err
