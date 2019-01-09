"""PyTorch-compatible transforms and collate functions for ASR.

This module demonstrates how to create PyTorch-compatible datasets for speech
recognition using `audlib`.
"""

import numpy as np
from scipy.signal import lfilter
from scipy.fftpack import idct
import torch
from torch.utils.data.dataloader import _use_shared_memory

from audlib.sig.window import hamming
from audlib.sig.fbanks import MelFreq
from audlib.sig.stproc import stana, numframes

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Transform functions below


class Melspec(object):
    """Compute MFCC of speech samples drawn from Dataset."""

    def __init__(self, sampling_rate, window_length=.0256, frame_rate=100,
                 nfft=512, nmel=40):
        """Initialize Mel filterbanks."""
        super(Melspec, self).__init__()
        self.sr = sampling_rate
        self.hopsize = int(sampling_rate/frame_rate)
        self.wind = hamming(int(window_length*sampling_rate), hop=self.hopsize)
        self.melbank = MelFreq(sampling_rate, nfft, nmel, unity=True)
        self.nmel = nmel

    def __call__(self, sample):
        """Extract MFCCs of signals in sample.

        Assume sample = {'data': signal,
                         'label': label sequence
                         }
        """
        sig = sample['data']
        label = sample['label']

        # pre-emphasis
        sig = lfilter([1, -0.97], [1], sig)

        # No dithering

        # extract MFCCs
        nframe = numframes(sig, self.sr, self.wind,
                           self.hopsize, trange=(0, None))
        mfcc = np.zeros((nframe, self.nmel))
        for ii, frame in enumerate(stana(sig, self.sr, self.wind, self.hopsize,
                                         trange=(0, None))):
            mfcc[ii] = self.melbank.mfcc(frame)

        # cepstral mean normalization
        mfcc -= mfcc.mean(axis=0)

        # inverse transform to mel spectral domain
        melspec = idct(mfcc, norm='ortho')

        out = {'data': melspec, 'label': label}

        return out


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

        Assume sample = {'data': feature in ndarray,
                         'label': label sequence in list of int
                         }
        """
        feat, label = sample['data'], sample['label']
        if self.train:
            input = self.append_token(label)
            output = self.append_token(label, input=False)
        else:
            input = np.zeros(1)
            output = np.zeros(1)

        return feat, input, output


# Collate functions below


def my_collate_fn(batch):
    # TODO: Need to work on docstring.
    '''
    sort sequences and seq_lengths in the batch according to sequence length
    sort inputs, outputs and label_lengths in the batch according to character level length
    give character level length sorting index to resort the sequence after unpacked in decoder

    B for batchsize, S for sequence length (frames), D for dimension (of frequency), L for character level length

    sorted according to sequence length:
    seq_sorted [B, S, D]
    length_sorted [B]

    sorted according to character level length (label_length):
    inputs_sorted [B, L]
    outputs_concat [sum of valid label length]
    label_length_sorted [B]
    lbl_perm_idx [B]
    '''
    batch_size = len(batch)

    max_len = 0
    max_output_len = 0
    sum_output_len = 0
    for sequence, input, output in batch:
        # REVIEW: @Raymond changed the interface here. @Shangwu Make sure this looks good.
        seq_length, upper_seq_length = len(sequence), len(sequence)
        label_length = len(input)
        max_len = max(max_len, upper_seq_length)
        max_output_len = max(max_output_len, label_length)
        sum_output_len += label_length

    sequences = None
    seq_lengths = None
    inputs = None
    outputs = None
    outputs_concat = None
    label_lengths = None
    unshuffle_idx = None
    if _use_shared_memory:
        sequences = torch.FloatStorage._new_shared(
            batch[0][0], batch_size*max_len*40).new(batch_size, max_len, 40).zero_().float()
        seq_lengths = torch.FloatStorage._new_shared(
            batch[0][0], batch_size).new(batch_size,).zero_().int()
        inputs = torch.FloatStorage._new_shared(
            batch[0][0], batch_size*max_output_len).new(batch_size, max_output_len).zero_().long()
        outputs = torch.FloatStorage._new_shared(
            batch[0][0], batch_size*max_output_len).new(batch_size, max_output_len).zero_().long()
        outputs_concat = torch.FloatStorage._new_shared(
            batch[0][0], sum_output_len).new(sum_output_len, ).zero_().long()
        label_lengths = torch.FloatStorage._new_shared(
            batch[0][0], batch_size).new(batch_size,).zero_().int()
        unshuffle_idx = torch.FloatStorage._new_shared(
            batch[0][0], batch_size).new(batch_size,).zero_().long()
    else:
        sequences = batch[0][0].new(batch_size, max_len, 40).zero_().float()
        seq_lengths = batch[0][0].new(batch_size,).zero_().int()
        inputs = batch[0][0].new(batch_size, max_output_len).zero_().long()
        outputs = batch[0][0].new(batch_size, max_output_len).zero_().long()
        outputs_concat = batch[0][0].new(sum_output_len, ).zero_().long()
        label_lengths = batch[0][0].new(batch_size,).zero_().int()
        unshuffle_idx = batch[0][0].new(batch_size,).zero_().long()

    i = 0
    for sequence, input, output in batch:
        # REVIEW: @Raymond changed the interface here. @Shangwu Make sure this looks good.
        seq_length, upper_seq_length = len(sequence), len(sequence)
        label_length = len(input)
        sequences[i, :seq_length, :] = sequence

        inputs[i, :label_length] = input
        outputs[i, :label_length] = output

        seq_lengths[i] = upper_seq_length  # upper_seq_length
        label_lengths[i] = label_length
        i += 1

    return sequences, seq_lengths, inputs, outputs, label_lengths
