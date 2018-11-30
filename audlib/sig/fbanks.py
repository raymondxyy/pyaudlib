"""FilterBANKS for audio analysis and synthesis.

#TODO:
    - [-] LinFreq: Linear-frequency filterbank (STFT filterbank view)
    - [-] MelFreq: Mel-frequency filterbank
    - [ ] Gammatone: Gammatone filterbank

"""

import numpy as np
from numpy.fft import rfft, fft
from scipy.fftpack import dct
from scipy import signal

from .auditory import hz2mel, mel2hz


class Filterbank(object):
    """An abstract class of filterbanks.

    All types of filterbanks should subclass this class and implement:
        * __len__(): number of filterbanks
        * __get_item__(i): i-th filter object
        * freqz(i): frequency response of i-th filter
        * filter(sig, i): filter signal `sig` by i-th filter
    """

    def __len__(self):
        """Return the number of frequency channels."""
        raise NotImplementedError

    def __getitem__(self, k):
        """Obtain k-th filter from the filterbank."""
        raise NotImplementedError

    def freqz(self, k):
        """Frequency response of the k-th filter."""
        raise NotImplementedError

    def filter(self, sig, k):
        """Filter a signal through k-th filter."""
        raise NotImplementedError


class LinFreq(Filterbank):
    """The linear-frequency filterbank.

    This class is implemented using the STFT bandpass filter view.
    """

    def __init__(self, wind, dsamp=True):
        """Create a bank of bandpass filters using prototype lowpass window.

        Parameters
        ----------
        wind : array_like
            A window function.

        """
        self.nchan = len(wind)  # at least N channels for N-point window
        self.wind = wind
        self.filts = np.zeros((self.nchan, len(wind)), dtype=np.complex_)
        for k in range(self.nchan):  # make bandpass filters
            wk = 2*np.pi*k / self.nchan
            self.filts[k] = wind * np.exp(1j*wk*np.arange(len(wind)))

    def __len__(self):
        """Return number of filters."""
        return self.nchan

    def __getitem__(self, k):
        """Return k-th channel FIR filter coefficients."""
        return self.filts[k]

    def freqz(self, k, nfft=None):
        """Return frequency response of k-th channel filter."""
        if nfft is None:
            nfft = max(1024, int(2**np.ceil(np.log2(self.nchan)))
                       )  # at least show 1024 frequency points
        ww = np.linspace(0, 2, num=nfft, endpoint=False)
        hh = fft(self.filts[k], n=nfft)
        return ww, hh

    def filter(self, sig, k):
        """Filter signal by k-th filter."""
        demod = np.exp(-1j*(2*np.pi*k/self.nchan)*np.arange(len(sig)))
        return signal.fftconvolve(sig, self.filts[k], 'same') * demod


class MelFreq(Filterbank):
    """The Mel-frequency filterbank."""

    def __init__(self, sr, nfft, nchan, flower=0., fupper=.5, unity=False):
        """Construct a Mel filterbank.

        Parameters
        ----------
        sr : int or float
            Sampling rate
        nfft : int
            DFT size
        nchan : int
            Number of filters in filterbank
        flower : int or float <0.>
            Lowest center-frequency in filterbank. Could either be in terms of
            Hz or 2*pi. Default to 0. (DC).
        fupper : int or float <.5>
            Higest center-frequency in filterbank. Could either by in terms of
            Hz or 2*pi. Default to .5 (Nyquist).

        Returns
        -------

        """
        self.nfft = nfft
        self.nchan = nchan
        # Find frequency (Hz) endpoints
        if flower > 1:  # assume in Hz
            hzl = flower
        else:  # assume in normalized frequency
            hzl = flower * sr

        if fupper > 1:
            hzh = fupper
        else:
            hzh = fupper * sr

        # Calculate mel-frequency endpoints
        mfl = hz2mel(hzl)
        mfh = hz2mel(hzh)

        # Calculate mel frequency range `mfrng`
        # Calculate mel frequency increment between adjacent channels `mfinc`
        mfrng = mfh - mfl
        mfinc = mfrng * 1. / (nchan+1)
        mfc = mfl + mfinc * np.arange(1, nchan+1)  # mel center frequencies

        # Calculate the DFT bins for [fl[0], fc[0], fc[P-1], fh[P-1]
        # p+1 markers for p channels
        dflim = mel2hz(
            mfl + mfinc*np.array([0, 1, nchan, nchan+1])) / sr * nfft
        dfl = int(dflim[0])+1  # lowest DFT bin required
        dfh = min(nfft//2, int(dflim[-1])-1)  # highest DFT bin required

        # Map all useful DFT bins to mel-frequency centers
        mfc = (hz2mel(sr * np.arange(dfl, dfh+1) * 1. / nfft)-mfl) / mfinc
        if mfc[0] < 0:
            mfc = mfc[1:]
            dfl += 1
        if mfc[-1] >= nchan+1:
            mfc = mfc[:-1]
            dfh -= 1
        mfc_fl = np.floor(mfc)
        mfc_ml = mfc - mfc_fl  # multiplier for upper filter

        df2 = np.argmax(mfc_fl > 0)
        df3 = len(mfc_fl) - np.argmax(mfc_fl[::-1] < nchan)
        df4 = len(mfc_fl)
        row = np.concatenate((mfc_fl[:df3], mfc_fl[df2:df4]-1))
        col = np.concatenate((range(df3), range(df2, df4)))
        val = np.concatenate((mfc_ml[:df3], 1-mfc_ml[df2:df4]))

        # Finally, cache values for each filter
        self.filts = []
        for ii in range(self.nchan):
            idx = row == ii
            if unity:
                self.filts.append((col[idx]+dfl, val[idx]/sum(val[idx])))
            else:
                self.filts.append((col[idx]+dfl, val[idx]))

    def __len__(self):
        """Return the number of frequency channels."""
        return self.nchan

    def __getitem__(self, k):
        """Obtain k-th filter from the filterbank."""
        return self.filts[k]

    def freqz(self, k):
        """Frequency response of the k-th filter."""
        ww = np.arange(self.nfft//2+1)/self.nfft*2
        hh = np.zeros(self.nfft//2+1)
        dfb, val = self.filts[k]
        hh[dfb] = val
        return ww, hh

    def filter(self, sig, k):
        """Filter a signal through k-th filter."""
        dfb, val = self.filts[k]
        dft_sig = rfft(sig, self.nfft)
        return val.dot(dft_sig[dfb])

    def melspec(self, sig):
        """Return mel log spectrum of a signal."""
        out = np.zeros(self.nchan)
        for kk in range(self.nchan):
            dfb, val = self.filts[kk]
            out[kk] = val.dot(np.abs(rfft(sig, self.nfft)[dfb])**2)
        return np.log(out)

    def mfcc(self, sig):
        """Return mel-frequency cepstral coefficients (MFCC)."""
        return dct(self.melspec(sig), norm='ortho')