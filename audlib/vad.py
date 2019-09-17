"""Methods for Voice (or Speech) Activity Detection."""
import numpy as np
from scipy.signal import lfilter

from .sig.stproc import numframes, stana
from .sig.temporal import zcrate, lpc_parcor, lpcerr


class ZCREnergy(object):
    """Voice activity detection by zero-crossing rate and energy.

    This class implements "Separation of Voiced and Unvoiced using Zero
    crossing rate and Energy of the Speech Signal" by Bachu et al.
    """

    def __init__(self, sr, wind, hop):
        """Instantiate a ZCREnergy class."""
        super(ZCREnergy, self).__init__()
        self.numframes = lambda sig: numframes(sig, sr, wind, hop, center=True)
        self.stana = lambda sig: stana(sig, sr, wind, hop, center=True)

    def zcre(self, sig, lpc_order=0):
        """Compute the ratio of zero-crossing rate and energy of a signal."""
        out = np.empty((self.numframes(sig), 2))
        for ii, frame in enumerate(self.stana(sig)):
            if lpc_order:
                alphas, _ = lpc_parcor(frame, lpc_order)
                frame = lpcerr(frame, alphas)
            out[ii, 0] = zcrate(frame)
            out[ii, 1] = np.sum(frame**2)

        out = out - out.mean(axis=0)
        out = out / out.std(axis=0)
        return out


class SpectralEnergy(object):
    """A spectral-energy-based voice activity detector.

    This simple energy-based VAD operates on short-time power spectra with a
    linear frequency scale (normal STFT).

    """

    def __init__(self, sr, nfft, fmin=300., fmax=4000.):
        """Instantiate a VAD class.

        Keyword Parameters
        ------------------
        fmin: float, 300
            Minimum frequency from which energy is collected.
        fmax: float, 4000
            Maximum frequency from which energy is collected.

        """
        assert (fmin > 0) and (fmin < fmax) and (fmax < sr/2)
        fintv = sr / nfft
        self.bmin = int(fmin//fintv)
        self.bmax = int(min(fmax//fintv, nfft//2))  # inclusive
        self.nfft = nfft

    def __call__(self, pspec, dbfloor=-30., smoothframes=0):
        """Detect speech-active frames from a power spectra.

        Keyword Parameters
        ------------------
        dbfloor: float, -30
            Energy floor for a frame below maximum energy to be considered speech.
        smoothframes: int, 0
            Number of frames to apply a moving-average filter on power contour.

        """
        assert pspec.shape[1] == (self.nfft//2+1), "Incompatible dimension."
        pspec = pspec[:, self.bmin:self.bmax+1].sum(axis=1)
        if smoothframes > 0:
            pspec = lfilter(1/smoothframes * np.ones(smoothframes), 1, pspec)
        return pspec > (10**(dbfloor/10))*pspec.max()

