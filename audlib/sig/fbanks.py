"""FilterBANKS for audio analysis and synthesis."""
import math

import numpy as np
from numpy.fft import rfft, fft
from scipy.fftpack import dct

from .auditory import hz2mel, mel2hz
from .auditory import erb_space, erb_filters, erb_fbank, erb_freqz
from .window import hamming
from .temporal import convdn, conv
from .spectral import logmag


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
        self.wgts = np.zeros((nfft//2+1, nchan))
        for ii in range(self.nchan):
            idx = row == ii
            if unity:
                dftbin, dftwgt = col[idx]+dfl, val[idx]/sum(val[idx])
            else:
                dftbin, dftwgt = col[idx]+dfl, val[idx]
            self.filts.append((dftbin, dftwgt))
            self.wgts[dftbin, ii] = dftwgt

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

    def melspec(self, powerspec):
        """Return the mel spectrum of a signal."""
        return powerspec @ self.wgts

    def mfcc(self, powerspec, mean_norm=True):
        """Return mel-frequency cepstral coefficients (MFCC)."""
        cep = dct(logmag(self.melspec(powerspec)), norm='ortho')
        if mean_norm:
            cep -= cep.mean(axis=0)
        return cep


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
        reverse: bool, True
           Index frequency in the reverse direction so that the default goes
           from low to high.

        """
        super(Gammatone, self).__init__()
        self.sr = sr
        self.num_chan = num_chan
        if center_frequencies is None:
            self.cf = erb_space(num_chan, 100., sr/2)[::-1]
        elif len(center_frequencies) == num_chan:
            self.cf = center_frequencies
        else:
            assert len(center_frequencies) == 2,\
                "Fail to interpret center frequencies!"
            self.cf = erb_space(num_chan, *center_frequencies)[::-1]

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

    def freqz(self, k, nfft=1024, powernorm=False):
        """Compute k-th channel's frequency reponse.

        Parameters
        ----------
        k: int
            ERB frequency channel.
        nfft: int, None
            Number of linear frequency points.
        powernorm: bool, False
            Normalize power to unity if True.
        """
        ww, hh = erb_freqz(*self.filters[k], nfft)
        if powernorm:
            hh /= sum(hh.real**2 + hh.imag**2)

        return ww, hh

    def filter(self, sig, k, cascade=True):
        """Filter signal with k-th channel."""
        return erb_fbank(sig, *self.filters[k], cascade=cascade)

    def gammawgt(self, nfft, powernorm=False, squared=True):
        """Return the Gammatone weighting function for STFT.

        Parameters
        ----------
        nfft: int
            Number of DFT points.
        powernorm: bool, False
            Normalize power of Gammatone weighting function to unity.
        squared: bool, True
            Apply squared Gammtone weighting.
        """
        wts = np.empty((nfft//2+1, self.num_chan))
        for k in range(self.num_chan):
            wts[:, k] = np.abs(self.freqz(k, nfft, powernorm)[1][:nfft//2+1])

        if squared:
            wts = wts**2

        return wts


class ConstantQ(Filterbank):
    """Direct implementation of Judith Brown's Constant Q transform (CQT)."""

    def __init__(self, sr, fmin, bins_per_octave=12, fmax=None, nchan=None,
                 zphase=True):
        """Instantiate a constant Q transform class.

        Parameters
        ----------
        sr: int or float
            Sampling rate.
        fmin: int or float
            Lowest center frequency of the filterbank.
            Note that all other center frequencies are derived from this.
        bins_per_octave: int
            Number of bins per octave (double frequency).
            Default to 12, which corresponds to one semitone.
        fmax: int or float
            Highest center frequency of the filterbank.
            Default to None, which assumes Nyquist. If `nchan` is set, `fmax`
            will be ignored.
        nchan: int
            Total number of frequency bins.
            Default to None, which is determined from other parameters. If set,
            `fmax` will be adjusted accordingly.
        zphase: bool
            Center each window at time 0 rather than (Nk-1)//2. This is helpful
            for mitigating the effect of group delay at low frequencies.
            Default to yes.

        """
        assert fmin >= 100, "Small center frequencies are not supported."
        if nchan:  # re-calculate fmax
            self.nchan = nchan
            fmax = fmin * 2**(nchan / bins_per_octave)
            assert fmax <= sr/2,\
                "fmax exceeds Nyquist! Consider reducing nchan or fmin."
            assert nchan == math.ceil(bins_per_octave*np.log2(fmax/fmin))
        else:
            fmax = fmax if fmax else sr/2
            self.nchan = math.ceil(bins_per_octave * np.log2(fmax/fmin))
        self.sr = sr
        self.qfactor = 1 / (2**(1/bins_per_octave) - 1)
        self.cfs = fmin * 2**(np.arange(self.nchan)/bins_per_octave)  # fcs
        self.zphase = zphase
        self.filts = []
        for ii, k in enumerate(range(self.nchan)):  # make bandpass filters
            cf = self.cfs[ii]
            wk = 2*np.pi*cf / sr
            wsize = math.ceil(self.qfactor*sr/cf)
            if zphase and (wsize % 2 == 0):  # force odd-size window for 0phase
                wsize += 1
            if zphase:
                mod = np.exp(1j*wk*np.arange(-(wsize-1)//2, (wsize-1)//2 + 1))
            else:
                mod = np.exp(1j*wk*np.arange(wsize))
            wind = hamming(wsize)
            self.filts.append(wind/wind.sum() * mod)

    def __len__(self):
        """Return number of filters."""
        return self.nchan

    def __getitem__(self, k):
        """Return k-th channel FIR filter coefficients."""
        return self.filts[k]

    def freqz(self, k, nfft=None):
        """Return frequency response of k-th channel filter."""
        if nfft is None:
            nfft = max(1024, int(2**np.ceil(np.log2(len(self.filts[k]))))
                       )  # at least show 1024 frequency points
        ww = 2*np.pi * np.arange(nfft)/nfft
        hh = fft(self.filts[k], n=nfft)
        return ww, hh

    def filter(self, sig, k, fr=None, zphase=True):
        """Filter signal by k-th filter."""
        wk = 2*np.pi*self.cfs[k] / self.sr
        decimate = int(self.sr/fr) if fr else None
        if decimate:
            demod = np.exp(-1j*wk*np.arange(0, len(sig), decimate))
            return convdn(sig, self.filts[k], decimate,
                          zphase=zphase)[:len(demod)] * demod
        else:
            demod = np.exp(-1j*wk*np.arange(len(sig)))
            return conv(sig, self.filts[k], zphase=zphase)[:len(sig)] * demod

    def cqt(self, sig, fr):
        """Return the constant Q transform of the signal.

        Parameters
        ----------
        sig: array_like
            Signal to be processed.
        fr: int
            Frame rate (or SR / hopsize in seconds) in Hz.

        """
        decimate = int(self.sr/fr)  # consistent with filter definition
        out = np.empty((self.nchan, math.ceil(len(sig)/decimate)),
                       dtype='complex_')
        for kk in range(self.nchan):
            out[kk] = self.filter(sig, kk, fr=fr)

        return out.T
