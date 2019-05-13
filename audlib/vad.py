"""Methods for Voice (or Speech) Activity Detection."""
import numpy as np

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
