"""Dataset Classes for Speech Enhancement Applications."""

from random import randint, randrange

import numpy as np

from ..io.batch import lsfiles
from ..io.audio import audioinfo, audioread, no_shorter_than
from ..sig.util import add_noise
from .dataset import Dataset, SEDataset
from .datatype import Audio, NoisySpeech


def randsel(audobj, minlen=0, maxlen=None, unit="second"):
    """Randomly select a portion of audio from path.

    Parameters
    ----------
    path: str
        File path to audio.
    minlen: float, optional
        Inclusive minimum length of selection in seconds or samples.
    maxlen: float, optional
        Exclusive maximum length of selection in seconds or samples.
    unit: str, optional
        The unit in which `minlen` and `maxlen` are interpreted.
        Options are:
            - 'second' (default)
            - 'sample'

    Returns
    -------
    tstart, tend: tuple of int
        integer index of selection

    """
    if type(audobj) is np.ndarray:
        assert unit == "sample", "ndarray does not have sampling rate!"
        sigsize = len(audobj)
        minoffset = int(minlen)
        maxoffset = sigsize if not maxlen else int(maxlen)
    else:
        info = audioinfo(audobj)
        sr, sigsize = info.samplerate, info.frames
        if unit == 'second':
            minoffset = int(minlen*sr)
            maxoffset = int(maxlen*sr) if maxlen else sigsize
        else:
            minoffset = minlen
            maxoffset = maxlen if maxlen else sigsize

    assert (minoffset < maxoffset) and (minoffset < sigsize), \
        f"""BAD: siglen={sigsize}, minlen={minoffset}, maxlen={maxoffset}"""

    # Select begin sample
    tstart = randrange(max(1, sigsize-minoffset))
    tend = randrange(tstart+minoffset, min(tstart+maxoffset, sigsize))

    return tstart, tend


def randread(fpath, sr=None, minlen=None, maxlen=None, unit='second'):
    """Randomly read a portion of audio from file."""
    nstart, nend = randsel(fpath, minlen, maxlen, unit)
    return audioread(fpath, sr=sr, start=nstart, stop=nend)


class RandSample(Dataset):
    """Create a dataset by random sampling of all valid audio files."""

    @staticmethod
    def isaudio(path):
        return path.endswith(('.wav', '.flac', '.sph'))

    def __init__(self, root, sr=None, filt=None,
                 minlen=None, maxlen=None,
                 unit='second',
                 transform=None,
                 cache=False):
        """Instantiate a random sampling dataset.

        Parameters
        ----------
        root: str
            Dataset root directory.
        sr: int, optional
            Forced sampling rate. Default to None, which accepts any rate.
        filt: callable, optional
            A function to decide if a file path should be accepted or not.
            Default to None, which accepts .wav, .flac, and .sph.
        minlen: float or int, optional
            Minimum duration of each sample in seconds or samples to be read.
            Default to None, which means unconstrained lengths.
        maxlen: float or int, optional
            Maximum duration of each sample in seconds or samples to be read.
            Default to None, which means unconstrained lengths.
        unit: str, optional
            The unit in which `minlen` and `maxlen` are interpreted.
            Options are:
                - 'second' (default)
                - 'sample'
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
        self.minlen, self.maxlen = minlen, maxlen
        self.unit = unit
        self.transform = transform
        self._cached = [] if cache else None

        self._filepaths = lsfiles(
            self.root, lambda p: (filt(p) if filt else self.isaudio(p))
            and no_shorter_than(p, self.minlen, unit=self.unit))

        if cache:
            for path in self._filepaths:
                self._cached.append(audioread(path, sr=self.sr)[0])

            def _randsel(data):
                return randsel(data, minlen, maxlen, unit='sample')
            self.select = _randsel
        else:
            def _randread(p):
                return randread(p, sr, minlen, maxlen, unit)[0]
            self.read = _randread

    @property
    def filepaths(self):
        """Retrieve all file paths."""
        return self._filepaths

    def __getitem__(self, idx):
        """Get idx-th sample."""
        if self._cached is not None:
            nstart, nend = self.select(self._cached[idx])
            sample = Audio(self._cached[idx][nstart:nend], self.sr)
        else:
            sample = Audio(self.read(self.filepaths[idx]), self.sr)

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        """Return number of samples."""
        return len(self._filepaths)


class Additive(SEDataset):
    """ADDITIVE mixing of two datasets.

    One dataset is treated as a target signal dataset, while the other is
    treated as an interfering signal dataset. An example use of this class is
    generating synthetic noisy speech for training/testing deep-learning-based
    speech enhancement systems.
    """

    def __init__(self, targetset, noiseset,
                 snrs=[-20, -15, -10, -5, 0, 5, 10, 15, 20],
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
            Default to [-20, -15, -10, -5, 0, 5, 10, 15, 20].

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
        assert samp_clean.samplerate == samp_noise.samplerate,\
            "Inconsistent sampling rate!"
        snr = self.snrs[randint(0, len(self.snrs)-1)]

        samp_noisy = Audio(add_noise(samp_clean.signal, samp_noise.signal,
                           snr=snr), samp_clean.samplerate)

        sample = NoisySpeech(samp_noisy, samp_clean, samp_noise, snr)

        if self.transform:
            sample = self.transform(sample)

        return sample
