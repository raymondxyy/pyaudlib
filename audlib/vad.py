"""Methods for Voice (or Speech) Activity Detection."""
import numpy as np
from scipy.signal import lfilter

from .sig.stproc import numframes, stana
from .sig.temporal import zcrate, lpc_parcor, lpcerr
from .enhance import priori2filt


class ZCREnergy(object):
    """Voice activity detection by zero-crossing rate and energy.

    This class implements "Separation of Voiced and Unvoiced using Zero
    crossing rate and Energy of the Speech Signal" by Bachu et al.
    """

    def __init__(self, sr, wind, hop):
        """Instantiate a ZCREnergy class."""
        super(ZCREnergy, self).__init__()
        self.numframes = lambda sig: numframes(sig, wind, hop, center=True)
        self.stana = lambda sig: stana(sig, wind, hop, center=True)

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

    def __init__(self, mask=None):
        """Instantiate a VAD based on spectral energy with frequency mask.

        Keyword Parameters
        ------------------
        mask: list or numpy.ndarray, None
            Frequency indices that will be included for energy calculation.

        """
        self.mask = mask

    def __call__(self, pspec, dbfloor=-30., smoothframes=0):
        """Detect speech-active frames from a power spectra.

        Keyword Parameters
        ------------------
        dbfloor: float, -30
            Energy floor for a frame below maximum energy to be voiced.
        smoothframes: int, 0
            Number of frames to apply a moving-average filter on power contour.

        """
        if self.mask is not None:
            pspec = pspec[:, self.mask].sum(axis=1)
        else:
            pspec = pspec.sum(axis=1)
        if smoothframes > 0:
            pspec = lfilter(1/smoothframes * np.ones(smoothframes), 1, pspec)
        return pspec > (10**(dbfloor/10))*pspec.max()


def asnr_spec(noisyspec, snrsmooth=.98, noisesmooth=.98, llkthres=.15,
              rule='wiener'):
    # TODO: Test and improve this.
    """Implement the a-priori SNR estimation described by Scalart and Filho.

    This is very similar to `asnr`, except it's computed directly on noisy
    magnitude spectra instead of time-domain signals.

    Outputs
    -------
    xfilt: ndarray
        Filtered magnitude spectrogram.
    priori: ndarray
        A priori SNR.
    posteri: ndarray
        A posteriori SNR.
    vad: ndarray
        Speech-presence log likelihood ratio.

    """
    # Estimate initial noise PSD
    npsd = np.mean(noisyspec[:6, :]**2, axis=0)
    vad = np.empty(len(noisyspec))
    for ii, xspec in enumerate(noisyspec):
        xpsd = np.abs(xspec)**2
        posteri = xpsd / npsd
        posteri_prime = np.maximum(posteri - 1, 0)  # half-wave rectify
        if ii == 0:  # initialize priori SNR
            priori = snrsmooth + (1-snrsmooth)*posteri_prime
        else:
            priori = snrsmooth*(priori2filt(priori, rule)**2)\
                * posteri + (1-snrsmooth)*posteri_prime
        # compute speech presence log likelihood
        llk = posteri*priori / (1+priori) - np.log(1+priori)
        vad[ii] = np.mean(llk)
        if vad[ii] < llkthres:
            # noise only frame found, update Pn
            npsd = noisesmooth*npsd + (1-noisesmooth)*xpsd

    return vad >= llkthres
