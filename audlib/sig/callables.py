"""Common transforms refactored as CALLABLES."""
import numpy as np
import scipy.signal as signal

from .spectemp import pncc, modspec, invspec
from .spectral import logpow
from .window import hamming
from .transform import stft, istft
from .fbanks import Gammatone, ConstantQ, MelFreq


class STFT(object):
    """Short-time Fourier transform and its inverse."""
    def __init__(self, sr, windowlen, hop, nfft, center=True, synth=False,
                 zphase=False):

        self.wind = hamming(int(windowlen*sr), hop=hop, synth=synth)
        self.hop = hop
        self.sr = sr  # preserved for other classes
        self.nfft = nfft
        self.center = center
        self.synth = synth
        self.zphase = zphase

    def forward(self, sig):
        return stft(sig, self.wind, self.hop, self.nfft, self.center,
                    self.synth, self.zphase)

    def inverse(self, spec):
        return istft(spec, self.wind, self.hop, self.nfft, self.zphase)

    def __call__(self, sig):
        """Return the power spectra."""
        spec = self.forward(sig)
        return spec.real**2 + spec.imag**2


class CQT(object):
    """Constant-Q transform."""
    def __init__(self, sr, fr, fc_min, bins_per_octave):
        self.fbank = ConstantQ(sr, fc_min, bins_per_octave)
        self.fr = fr

    def __call__(self, sig):
        """Return the log power spectra."""
        spec = self.fbank.cqt(sig, self.fr)
        return spec.real**2 + spec.imag**2


class GammatoneSpec(object):
    """Wrapper for the spectrogram derived from the Gammatone filterbank.
    
    NOTE: this class does not implement the biquad filterbank, but rather a
    frequency integration approximation of it. See `GammatoneFB` for the actual
    filterbank.
    """
    def __init__(self, stftclass, nchan, powernorm=True, squared=True):
        self.stft = stftclass
        self.fbank = Gammatone(self.stft.sr, nchan)
        self.wts = self.fbank.gammawgt(self.stft.nfft, powernorm, squared)

    def forward(self, sig):
        spec = self.stft.forward(sig)
        return (spec.real**2 + spec.imag**2) @ self.wts

    def inverse(self, powerspec):
        return invspec(powerspec, self.wts)

    def __call__(self, sig):
        return self.forward(sig)


class GammatoneFB(object):
    """Wrapper for the Gammatone filterbank."""
    def __init__(self, sr, num_chan, center_frequencies=None):
        self.fbank = Gammatone(sr, num_chan, center_frequencies)
        self.sr = sr

    def __call__(self, sig, fr):
        deci = self.sr // fr
        return np.stack([
            self.fbank.filter(sig, k)[::deci] for k, _ in enumerate(self.fbank)
        ]).T


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
            return self.fbank.melspec(spec)
        return self.fbank.mfcc(spec, mean_norm=self.cmn)[:, :self.ncep]


class PNCC(object):
    """Spectral and cepstral features derived from the PNCC processing.
    
    See Also
    --------
    spectemp.pncc

    """
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
            return pspec * pncc(pspec, synth=True)
        return pncc(pspec, cmn=self.cmn, ccdim=self.ncep)


class ModSpec(object):
    """Wrapper for the modulation spectrogram by Kingsbury et al."""
    def __init__(self, sr, fr, nchan, fc_mod=6., norm=False, original=False):
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
        self.norm = norm
        self.original = original

    def __call__(self, sig):
        return modspec(sig, self.sr, self.fr, self.fbank,
                       self.lpf_env, self.lpf_mod, self.fc_mod,
                       self.norm, self.original)


class PNSpec(object):
    # TODO: Convert this class to a InverseSpec class
    """Short-time power-normalized spectra derived from PNCC."""
    def __init__(self, stftclass, nchan, full):
        """Instantiate a PNSpec class.

        Parameters
        ----------
        stftclass: class
            See STFT.
        nchan: int
            Number of frequency channels to keep.
        full: bool
            If False, return (T x nchan) mask derived from PNCC at each call.
            If True, return fullband mask by inverting the (T x nchan) mask.

        """
        raise NotImplementedError("PnSpec is obsolete.")
        self.stft = stftclass
        gtbank = Gammatone(self.stft.sr, nchan)
        self.wts = gtbank.gammawgt(self.stft.nfft, powernorm=True,
                                   squared=True)
        self.full = full

    def __call__(self, sig, fromspec=False):
        return self.forward(sig, fromspec)

    def pnccmask(self, pnspec, full=False):
        mask = pncc(pnspec, tempmask=True, synth=True)
        if full:
            return invspec(mask, self.wts)
        return mask

    def forward(self, sig, frompspec=False):
        if not frompspec:
            sig = self.stft.forward(sig)
            pspec = sig.real**2 + sig.imag**2
        else:
            pspec = sig
        pnspec = pspec @ self.wts
        if self.full:
            return pspec * self.pnccmask(pnspec, True)
        return pnspec * self.pnccmask(pnspec)

    def inverse(self, pnspec):
        """Invert a power-normalized spectra to a full-band power spectra."""
        if self.full:
            return pnspec
        return invspec(pnspec, self.wts)


class Compose(object):
    """Composes several transforms together.

    Copy from https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#Compose.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string