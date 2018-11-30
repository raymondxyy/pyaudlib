"""Synthetic noisy speech dataset with additive noise.

User should provide the clean speech dataset and noise dataset, respectively.
"""

import os
from random import randint

import numpy as np

from audlib.io.batch import dir2files
from audlib.io.audio import audioread

# Additive noise mixing SNRs
_snrs = [-np.inf, -20, -15, -10, -5, 0, 5, 10, 15, 20, np.inf]
_exts = ('.wav', '.sph', '.flac')
_norm = True


class Additive(object):
    """Generate a list of noisy-clean signal pairs by adding noise to speech.

    An example use of this class is generating synthetic noisy speech for
    training/testing deep-learning-based speech enhancement systems.

    Arguments
    ---------
    sdir: string
        Clean speech directory. Every single valid audio file in this dir will
        be processed.
    ndir: string
        Noise directory. Every valid audio file in this dir will be considered
        noise. For each speech sample taken, one noise sample will be randomly
        drawn from the pool and mixed with clean speech.

    Keyword Arguments
    -----------------
    snrs: list of int/float/np.inf
        SNRs to be randomly sampled.
        Default to [-inf, -20, -15, ..., 15, 20, +inf].
    exts: tuple of strings
        A list of accepted file extensions. Default to (.wav, .sph, .flac)
    """

    def __init__(self, sdir, ndir, snrs=_snrs, exts=_exts):
        """Build a synthetic dataset of noisy speech with additive noise."""
        self.sdir = sdir
        self.ndir = ndir
        assert os.path.exists(sdir), "Speech directory does not exist!"
        assert os.path.exists(ndir), "Noise directory does not exist!"

        self.slist = dir2files(sdir, lambda fn: fn.endswith(exts))
        self.nlist = dir2files(ndir, lambda fn: fn.endswith(exts))
        self.snrs = snrs

        # Assume every speech file is mixed with every noise file, at each SNR
        self._len = len(self.slist) * len(self.nlist) * len(self.snrs)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        # Exhaust all speech/noise/snr combinations in row-major order:
        # 0 --> s[0],n[0],snr[0]
        # 1 --> s[0],n[0],snr[1] ...
        # k --> s[0],n[1],snr[0] ...
        cleanpath = self.slist[idx // (len(self.nlist)*len(self.snrs))]
        noisepath = self.nlist[(idx % (len(self.nlist)*len(self.snrs))
                                ) // len(self.snrs)]
        snr = self.snrs[idx % len(self.snrs)]
        sigs, sr1 = audioread(cleanpath)
        sign, sr2 = audioread(noisepath)
        assert sr1 == sr2, 'Inconsistent sampling rate!'
        if len(sigs) > len(sign):  # repeat noise if shorter than speech
            sign = np.tile(sign, int(np.ceil(len(sigs)/len(sign))))
            sign = sign[:len(sigs)]
        elif len(sigs) < len(sign):  # truncate noise if too long
            sign = sign[:len(sigs)]

        noisy, clean, vad = self.mix(sigs, sign, sr1, snr)

        sample = {'sr': sr1,
                  'chan1': noisy,
                  'clean': clean,
                  'vad': vad,
                  'snr': snr}

        return sample

    def mix(self, sigs, sign, sr, snr):
        """Mix clean signal and noise."""
        noisy = np.zeros_like(sigs)
        clean = np.zeros_like(sigs)
        vad = []
        if snr == np.inf:  # pure speech
            noisy[:] = sigs
            clean[:] = noisy
            vad.append((0., len(sigs)*1./sr))
        elif snr == -np.inf:  # pure noise
            noisy[:] = sign
        else:  # numerical SNRs
            # Randomly choose if noise or speech spans entire range.
            # This will create transition of SNR:
            #   +inf --> finite SNR speech goes first
            #   -inf --> finite SNR noise goes first
            # make sure at least half of entire duration is interfered
            nstart = randint(0, len(sigs)//2)
            nend = randint(nstart+len(sigs)//2, len(sigs))
            if randint(0, 1):  # speech goes first
                noisy[:] = sigs
                # Compute segmental SNR and add noise
                se = np.sum(sigs[nstart:nend]**2)  # signal energy
                ne = np.sum(sign[nstart:nend]**2)  # noise energy
                scale = np.sqrt(se/ne / (10**(snr/10.)))
                noisy[nstart:nend] += sign[nstart:nend]*scale
                clean[:] = sigs
                vad.append((0, len(sigs)*1./sr))
            else:  # noise goes first
                noisy[:] = sign
                # Compute segmental SNR and add speech
                se = np.sum(np.abs(sigs[nstart:nend])**2)  # signal energy
                ne = np.sum(np.abs(sign[nstart:nend])**2)  # noise energy
                scale = np.sqrt(ne/se * (10**(snr/10.)))
                noisy[nstart:nend] += sigs[nstart:nend]*scale
                clean[nstart:nend] = sigs[nstart:nend]*scale
                vad.append((nstart*1./sr, nend*1./sr))

        return noisy, clean, vad


def test_additive():
    sdir = "/home/xyy/data/an4"  # change to your directory
    ndir = sdir  # for test purpose only
    dataset = Additive(sdir, ndir)
    sample = dataset[1]
    print("Number of instances: [{}]".format(len(dataset)))


if __name__ == '__main__':
    test_additive()
