"""PyTorch-compatible transforms and collate functions for ASR.

This module demonstrates how to create PyTorch-compatible datasets for speech
recognition using `audlib`.
"""

import numpy as np
from scipy.signal import lfilter
from scipy.fftpack import dct, idct

from audlib.sig.window import hamming
from audlib.sig.fbanks import MelFreq
from audlib.sig.transform import stpowspec


class Melspec(object):
    """Compute MFCC of speech samples drawn from Dataset."""

    def __init__(self, srate, wlen=.0256, frate=100, nfft=512, nmel=40):
        """Initialize Mel filterbanks."""
        super(Melspec, self).__init__()
        self.sr = srate
        self.melbank = MelFreq(srate, nfft, nmel, unity=True)
        self.nmel = nmel

        hop = int(srate/frate)
        wind = hamming(int(wlen*srate), hop=hop)

        def _melspec(sig):
            """Compute melspec after cepstral-mean-norm."""
            pspec = stpowspec(sig, wind, hop, nfft)
            melspec = pspec @ self.melbank.wgts
            smallpower = melspec < 10**(-8)  # -80dB power floor
            melspec[~smallpower] = np.log(melspec[~smallpower])
            melspec[smallpower] = np.log(10**(-8))
            mfcc = dct(melspec, norm='ortho')
            melspec = idct(mfcc-mfcc.mean(axis=0), norm='ortho')
            return melspec

        self._melspec = _melspec

    def __call__(self, sample):
        """Extract MFCCs of signals in sample.

        Assume sample is a SpeechTranscript class with the following fields:
            signal - signal samples
            samplerate - sampling rate in Hz
            label - integer list of labels
            transcript - actual transcript as a string
        """
        assert self.sr == sample.samplerate, "Incompatible sampling rate."
        sig = sample.signal

        # pre-emphasis
        sig = lfilter([1, -0.97], [1], sig)

        # No dithering

        # Compute melspec
        sample.signal = self._melspec(sig)

        return sample


class FinalTransform(object):
    """Interface with transforms and my_collate_fn."""

    def __init__(self, transmap, bos, eos, train=True):
        super(FinalTransform, self).__init__()
        assert (bos in transmap.vocabdict) and (eos in transmap.vocabdict)
        self.train = train
        self.tmap = transmap
        self.bos = bos
        self.eos = eos

    def __call__(self, sample):
        """Transform one data sample to its final form before collating.

        Assume sample is a SpeechTranscript class with the following fields:
            signal - (transformed) signal samples
            samplerate - sampling rate in Hz
            label - integer list of labels
            transcript - actual transcript as a string
        """
        feat, label = sample.signal, sample.label
        target = np.insert(label, len(label), self.tmap.vocabdict[self.eos])
        if self.train:
            inseq = np.insert(label, 0, self.tmap.vocabdict[self.bos])
        else:
            inseq = None

        return feat, inseq, target
