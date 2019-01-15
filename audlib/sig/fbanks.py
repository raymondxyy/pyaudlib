"""FilterBANKS for audio analysis and synthesis.

#TODO:
    - [-] LinFreq: Linear-frequency filterbank (STFT filterbank view)
    - [x] MelFreq: Mel-frequency filterbank
    - [-] Gammatone: Gammatone filterbank

"""

import numpy as np
from numpy.fft import rfft, fft
from scipy.fftpack import dct

from .auditory import hz2mel, mel2hz
from .auditory import erb_space, erb_filters, erb_fbank, erb_freqz


class Filterbank(object):
    """An abstract class of filterbanks.

    All types of filterbanks should subclass this class and implement:
        * __len__(): number of filterbanks
        * __getitem__(i): i-th filter object
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

    def __init__(self, wind, nchan=None):
        """Create a bank of bandpass filters using prototype lowpass window.

        Parameters
        ----------
        wind : array_like
            A window function.

        """
        self.nchan = (nchan if nchan is not None else len(wind))
        self.wind = wind
        self.nsym = (len(wind)-1) / 2.  # window point of symmetry
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
            nfft = max(1024, int(2**np.ceil(np.log2(len(self.wind))))
                       )  # at least show 1024 frequency points
        ww = 2*np.pi * np.arange(nfft)/nfft
        hh = fft(self.filts[k], n=nfft)
        return ww, hh

    def filter(self, sig, k):
        """Filter signal by k-th filter."""
        demod = np.exp(-1j*(2*np.pi*k/self.nchan)*np.arange(len(sig)))
        return np.convolve(sig, self.filts[k])[:len(sig)] * demod


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


class Gammatone(Filterbank):
    """The Gammatone filterbank."""

    def __init__(self, sr, num_chan, center_frequencies=None):
        """Instantiate a Gammatone filterbank.

        Parameters
        ----------
        sr: int
            Sampling rate.
        num_chan: int
            Number of frequency channels.
        center_frequencies: iterable, optional
            Center frequencies of each filter. There are 3 options:
            1. (Default) None. This sets f_lower to 100Hz, f_upper to Nyquist
            frequency, and assume equal spacing on linear frequency scale for
            other frequencies.
            2. Tuple of (`freqlower`, `frequpper`). This takes user-defined
            lower and upper bounds, and assume equal spacing on linear scale
            for other frequencies.
            3. Iterable of center frequencies. This allows every center
            frequency to be defined by user.

        """
        super(Gammatone, self).__init__()
        self.sr = sr
        self.num_chan = num_chan
        if center_frequencies is None:
            self.cf = erb_space(num_chan, 100., sr/2)
        elif len(center_frequencies) == num_chan:
            self.cf = center_frequencies
        else:
            assert len(center_frequencies) == 2,\
                "Fail to interpret center frequencies!"
            self.cf = erb_space(num_chan, *center_frequencies)

        self.filters = []
        for ii, cf in enumerate(self.cf):  # construct filter coefficients
            A0, A11, A12, A13, A14, A2, B0, B1, B2, gain = erb_filters(sr, cf)
            self.filters.append((A0, A11, A12, A13, A14, A2, B0, B1, B2, gain))

    def __len__(self):
        """Return number of channels."""
        return self.num_chan

    def __getitem__(self, k):
        """Get filter coefficients of k-th channel."""
        return self.filters[k]

    def freqz(self, k, nfft=None):
        """Compute k-th channel's frequency reponse."""
        if nfft is None:
            nfft = 1024
        return erb_freqz(*self.filters[k], nfft)

    def filter(self, sig, k):
        """Filter signal with k-th channel."""
        return erb_fbank(sig, *self.filters[k])
