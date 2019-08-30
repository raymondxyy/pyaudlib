"""PyTorch-compatible transforms and collate functions for ASR.

This module demonstrates how to create PyTorch-compatible datasets for speech
recognition using `audlib`.
"""

import numpy as np
from scipy.signal import lfilter
from scipy.fftpack import dct, idct
import torch
from torch.utils.data.dataloader import _use_shared_memory

from audlib.sig.window import hamming
from audlib.sig.fbanks import MelFreq
from audlib.sig.transform import stpowspec

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Transform functions below


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
            pspec = stpowspec(sig, srate, wind, hop, nfft)
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
    # TODO: Add docstring.
    # TODO: Possibly better function name.

    def __init__(self, transmap, train=True):
        super(FinalTransform, self).__init__()
        self.train = train
        self.tmap = transmap

    def append_token(self, label, input=True):
        # TODO: Add docstring.
        # TODO: Possibly better function name
        if input:  # prepend '&'
            return np.insert(label, 0, self.tmap.vocabdict['&'])
        else:  # append '*'
            return np.insert(label, len(label), self.tmap.vocabdict['*'])

    def __call__(self, sample):
        # TODO: Explain what each output is; consider changing interface.
        """Transform one data sample to its final form before collating.

        Assume sample is a SpeechTranscript class with the following fields:
            signal - (transformed) signal samples
            samplerate - sampling rate in Hz
            label - integer list of labels
            transcript - actual transcript as a string
        """
        feat, label = sample.signal, sample.label
        if self.train:
            input = self.append_token(label)
            output = self.append_token(label, input=False)
        else:
            input = np.zeros(1)
            output = np.zeros(1)

        return feat, input, output


# Collate functions below


def sort_by_feat_length_collate_fn(batch):
    """
    sort batch data according to feature length

    B for batch_size, F for feature length (frames), D for dimension (of
        frequency), L for character level length

    feats [B, S, D]
    feat_lengths [B]
    inputs [B, L]
    outputs [B, L]
    label_lengths [B]
    """

    # TODO: sort in collate_fn instead of network!!!

    batch_size = len(batch)

    max_len = 0
    max_output_len = 0
    sum_output_len = 0
    for feat, input, output in batch:
        feat_length = len(feat)
        label_length = len(input)
        max_len = max(max_len, feat_length)
        max_output_len = max(max_output_len, label_length)
        sum_output_len += label_length

    if _use_shared_memory:
        feats = torch.FloatStorage._new_shared(
            batch[0][0], batch_size*max_len*40).new(batch_size, max_len, 40).zero_().float()
        feat_lengths = torch.FloatStorage._new_shared(
            batch[0][0], batch_size).new(batch_size,).zero_().int()
        inputs = torch.FloatStorage._new_shared(
            batch[0][0], batch_size*max_output_len).new(batch_size, max_output_len).zero_().long()
        outputs = torch.FloatStorage._new_shared(
            batch[0][0], batch_size*max_output_len).new(batch_size, max_output_len).zero_().long()
        label_lengths = torch.FloatStorage._new_shared(
            batch[0][0], batch_size).new(batch_size,).zero_().int()
    else:
        feats = batch[0][0].new(batch_size, max_len, 40).zero_().float()
        feat_lengths = batch[0][0].new(batch_size,).zero_().int()
        inputs = batch[0][0].new(batch_size, max_output_len).zero_().long()
        outputs = batch[0][0].new(batch_size, max_output_len).zero_().long()
        label_lengths = batch[0][0].new(batch_size,).zero_().int()

    i = 0
    for feat, input, output in batch:
        feat_length = len(feat)
        label_length = len(input)
        feats[i, :feat_length, :] = feat

        inputs[i, :label_length] = input
        outputs[i, :label_length] = output

        feat_lengths[i] = feat_length
        label_lengths[i] = label_length
        i += 1

    return feats, feat_lengths, inputs, outputs, label_lengths
