"""Common transforms refactored as CALLABLE class."""
import numpy as np
import scipy.signal as signal

from .spectemp import pncc, modspec
from .spectral import logpow
from .window import hamming
from .transform import stft
from .fbanks import Gammatone, ConstantQ, MelFreq


class STFT(object):
    """Short-time Fourier transform."""
    def __init__(self, sr, windowlen, hop, nfft):
        def _stft(sig):
            return stft(sig, hamming(int(windowlen*sr), hop=hop), hop, nfft)

        self.forward = _stft
        self.sr = sr  # preserved for other classes
        self.nfft = nfft

    def __call__(self, sig):
        """Return the log power spectra."""
        spec = self.forward(sig)
        return logpow(spec)


class CQT(object):
    """Constant-Q transform."""
    def __init__(self, sr, fr, fc_min, bins_per_octave):
        self.fbank = ConstantQ(sr, fc_min, bins_per_octave)
        self.fr = fr

    def __call__(self, sig):
        """Return the log power spectra."""
        spec = self.fbank.cqt(sig, self.fr)
        return logpow(spec)


class GammatoneSpec(object):
    """Wrapper for the spectrogram derived from the Gammatone filterbank."""
    def __init__(self, stftclass, nchan):
        self.stft = stftclass
        self.fbank = Gammatone(self.stft.sr, nchan)
        self.wts = self.fbank.gammawgt(self.stft.nfft, powernorm=True,
                                       squared=True)

    def forward(self, sig):
        spec = self.stft.forward(sig)
        pspec = spec.real**2 + spec.imag**2
        return pspec @ self.wts

    def __call__(self, sig):
        return np.log(self.forward(sig).clip(1e-8))


class MFCC(object):
    """MFCC-related features."""
    def __init__(self, stftclass, nchan, ncep, cmn=True):
        """Instantiate a MFCC class.

        Parameters
        ----------
        stftclass: class
            See STFT.
        nchan: int
            Number of frequency channels to keep.
        ncep: int
            Number of cepstral dimensions to return.
            If 0, returns spectra instead (i.e. no DCT and nonlinearity).
        """
        self.stft = stftclass
        self.fbank = MelFreq(self.stft.sr, self.stft.nfft, nchan)
        self.ncep = ncep
        self.cmn = cmn

    def __call__(self, sig):
        spec = self.stft.forward(sig)
        spec = spec.real**2 + spec.imag**2
        if self.ncep == 0:
            return np.log(self.fbank.melspec(spec).clip(1e-8))
        return self.fbank.mfcc(spec, mean_norm=self.cmn)[:, :self.ncep]


class PNCC(object):
    """PNCC-related features."""
    def __init__(self, gammatonespec, ncep, cmn=True):
        """Instantiate a PNCC class.

        Parameters
        ----------
        stftclass: class
            See STFT.
        nchan: int
            Number of frequency channels to keep.
        ncep: int
            Number of cepstral dimensions to return.
            If 0, returns spectra instead (i.e. no DCT and nonlinearity).
        """
        self.gammatonespec = gammatonespec
        self.ncep = ncep
        self.cmn = cmn

    def __call__(self, sig):
        pspec = self.gammatonespec.forward(sig)
        if self.ncep == 0:
            return np.log((pspec * pncc(pspec, synth=True)).clip(1e-8))
        return pncc(pspec, cmn=self.cmn, ccdim=self.ncep)


class ModulationSpec(object):
    """Wrapper for the modulation spectrogram by Kingsbury et al."""
    def __init__(self, sr, fr, nchan, fc_mod=6., norm=False):
        """Instatiate a ModulationSpec class.

        Parameters
        ----------
        sr: int
            Sampling rate.
        fr: int
            Frame rate.
        fbank: fbanks.Filterbank object
            A Filterbank object that implements filter().
        fc_mod: float, 4.
            Modulation frequency of interest.
        norm: bool, False
            Whether to do long-term channel normalization.

        """
        self.sr = sr
        self.fr = fr
        self.fbank = Gammatone(sr, nchan)
        # TODO: give more options to user
        self.lpf_env = signal.firwin(501, 28/self.sr*2, window='hamming')
        self.lpf_mod = signal.firwin(25, 4/self.sr*2, window='hamming')
        self.fc_mod = fc_mod

    def __call__(self, sig):
        pspec = modspec(sig, self.sr, self.fr, self.fbank,
                        self.lpf_env, self.lpf_mod, self.fc_mod)
        return np.log(pspec.clip(1e-8))
