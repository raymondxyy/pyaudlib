"""Dataset Classes for Speech Enhancement Applications."""

from random import randint

import numpy as np

from ..io.batch import dir2files
from ..sig.util import additive_noise
from .dataset import Dataset, SEDataset
from .util import chk_duration, randsel, randread, audioread


class RandSample(Dataset):
    """Create a dataset by random sampling of all valid audio files."""

    def __init__(self, root, sr=None, mindur_per_file=None,
                 exts=('.wav', '.sph', '.flac'),
                 sampdur_range=(None, None),
                 transform=None,
                 cache=False):
        """Instantiate a random sampling dataset.

        Parameters
        ----------
        root: str
            Dataset root directory.
        sr: int, optional
            Forced sampling rate. Default to None, which accepts any rate.
        mindur_per_file: float, optional
            Minimum duration of each audio file in seconds. Shorter files
            will be ignored.
        exts: tuple of str, optional
            Accepted file extensions.
            Default to '.wav', '.sph', '.flac'.
        sampdur_range: tuple of float, optional
            Minimum and maximum duration of each sample in seconds to be read.
            Default to unconstrained lengths.
        transform: callable
            Tranform to be applied on samples.
        cache: bool
            Default only stores file paths and portion to read. If True,
            all signals will be read to memory in one shot, and indexing
            will be just slicing from the read arrays.

        """
        super(RandSample, self).__init__()
        self.root = root
        self.sr = sr
        self.mindur_per_file = mindur_per_file
        self.minlen, self.maxlen = sampdur_range
        self.exts = exts
        self.transform = transform
        self._cached = [] if cache else None

        self._all_files = dir2files(
            self.root, lambda path: path.endswith(exts)
            and chk_duration(path, minlen=self.mindur_per_file))

        if cache:
            for path in self._all_files:
                self._cached.append(audioread(path, sr=self.sr)[0])

    @property
    def all_files(self):
        """Retrieve all file paths."""
        return self._all_files

    def __getitem__(self, idx):
        """Get idx-th sample."""
        if self._cached is not None:
            maxlen = int(self.maxlen*self.sr) if self.maxlen else None
            nstart, nend = randsel(self._cached[idx],
                                   minlen=int(self.minlen*self.sr),
                                   maxlen=maxlen)
            data = self._cached[idx][nstart:nend]
        else:
            data, _ = randread(self.all_files[idx], sr=self.sr,
                               minlen=self.minlen, maxlen=self.maxlen)

        sample = {'data': data, 'sr': self.sr}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Additive(SEDataset):
    """ADDITIVE mixing of two datasets.

    One dataset is treated as a target signal dataset, while the other is
    treated as an interfering signal dataset. An example use of this class is
    generating synthetic noisy speech for training/testing deep-learning-based
    speech enhancement systems.
    """

    def __init__(self, targetset, noiseset,
                 snrs=[-np.inf, -20, -15, -10, -5, 0, 5, 10, 15, 20, np.inf],
                 transform=None):
        """Build a synthetic additive dataset.

        This class will mix a randomly chosen noise at a randomly chosen SNR
        into each signal in the target set.

        Parameters
        ----------
        targetset: Dataset class
            Target signal dataset. Assume each indexed sample consists of:
            {'sr': sampling rate, 'data': signal}
        noiseset: Dataset class
            Interfering signal dataset. Assume each indexed sample consists of:
            {'sr': sampling rate, 'data': signal}
        snrs: list of int/float/np.inf, optional
            SNRs to be randomly sampled.
            Default to [-inf, -20, -15, -10, -5, 0, 5, 10, 15, 20, +inf].

        See Also
        --------
        dataset.SEDataset

        """
        super(Additive, self).__init__()
        self.targetset = targetset
        self.noiseset = noiseset
        self.snrs = snrs
        self.transform = transform

    def __len__(self):
        """Each speech file is mixed with every noise file, at each SNR."""
        return len(self.targetset)

    def __getitem__(self, idx):
        """Retrieve idx-th sample."""
        samp_clean = self.targetset[idx]
        samp_noise = self.noiseset[randint(0, len(self.noiseset)-1)]
        snr = self.snrs[randint(0, len(self.snrs)-1)]

        sr, clean = samp_clean['sr'], samp_clean['data']
        noise = additive_noise(clean, samp_noise['data'], snr=snr)

        if snr == -np.inf:  # only output noise to simulate negative infinity
            chan1 = noise
            clean = np.zeros_like(clean)
        else:
            chan1 = clean + noise

        sample = {'chan1': {'sr': sr, 'data': chan1},
                  'clean': {'sr': sr, 'data': clean},
                  'noise': {'sr': sr, 'data': noise},
                  'snr': snr
                  }

        if self.transform:
            sample = self.transform(sample)

        return sample
