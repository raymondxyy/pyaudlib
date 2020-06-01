"""Some algorithms for PHASE estimation."""
import numpy as np
from numpy.fft import irfft

from .transform import stft
from .stproc import ola
from .spectral import magphasor


def griffin_lim(magspec, wind, hop, nfft, niter=32, init=None, zphase=True,
                reference=None):
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

    """
    we = (wind**2).sum()
    assert we > 0, "Window must not be complete zeros!"
    assert niter > 0, "Invalid number of iterations."

    # Init phase spectra
    if init is None:
        phasor = np.exp(1j*np.pi*(np.random.random_sample(magspec.shape)*2-1))
    elif type(init) in [float, int]:
        phasor = np.full_like(magspec, np.exp(1j*np.pi*init))
    else:
        assert magspec.shape == init.shape, "Incompatible init phase dimension."
        phasor = np.exp(1j*np.pi*init)

    # Define inverse STFT with weighted signal
    def istft_(spec):
        frames = irfft(spec, n=nfft)
        if zphase:
            fsize = len(wind)
            woff = (fsize-(fsize % 2)) // 2
            frames = np.concatenate((frames[:, (nfft-woff):],
                                     frames[:, :(fsize-woff)]), axis=1)
        else:
            frames = frames[:, :len(wind)]
        return ola(frames * wind / we, wind, hop)

    err = []
    for ii in range(niter):
        sig = istft_(magspec * phasor)
        _, phasor = magphasor(
            stft(sig, wind, hop, nfft, synth=True, zphase=zphase)
        )
        phasor = phasor[:len(magspec)]
        if reference is not None:
            err.append(((sig[:len(reference)]-reference)**2).mean())

    return phasor, err
