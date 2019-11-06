"""Dataset Classes for Speech Enhancement Applications."""
import os
from random import randint, randrange

from ..io.audio import audioread, no_shorter_than, randread
from ..sig.util import add_noise
from .dataset import AudioDataset, SEDataset
from .datatype import Audio, NoisySpeech


class RandSample(AudioDataset):
    """Create a dataset by random sampling of all valid audio files.

    This class sacrifices flexibility for a simple interface. If it doesn't
    cover your use case, you can use the more general dataset.AudioDataset.
    """

    @staticmethod
    def isaudio(path):
        return path.endswith(('.wav', '.flac', '.sph'))

    def __init__(self, root, minlen=0., maxlen=None, unit='second',
                 filt=None, transform=None, cache=False):
        """Instantiate a random sampling dataset.

        Parameters
        ----------
        root: str
            Dataset root directory.
        filt: callable, optional
            A function to decide if a file path should be accepted or not.
            Default to None, which accepts .wav, .flac, and .sph.
        read: callabel, optional
            Function to read in an audio file.
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
        self.minlen, self.maxlen = minlen, maxlen
        self.unit = unit
        self.transform = transform

        def _filt(path):
            """Filter based on audio lengths."""
            if filt:
                return filt(path) and no_shorter_than(path, minlen, unit)
            return self.isaudio(path) and no_shorter_than(path, minlen, unit)

        super(RandSample, self).__init__(root, filt=_filt)

        # Not recommended for large files
        if cache:
            self._cached = []
            for path in self._filepaths:
                self._cached.append(audioread(path))
        else:
            self._cached = None

    def __getitem__(self, idx):
        """Get idx-th sample, with the option to retreive cached sample."""
        if self._cached is not None:
            sig, sr = self._cached[idx]
            if self.unit == 'second':
                minoffset = int(self.minlen*sr)
                maxoffset = int(self.maxlen*sr) if self.maxlen else len(sig)
            else:
                minoffset = self.minlen
                maxoffset = self.maxlen if self.maxlen else len(sig)
            assert (minoffset < maxoffset) and (minoffset <= len(sig)), \
                f"""BAD: siglen={len(sig)}, minlen={minoffset},
                    maxlen={maxoffset}"""
            nstart = randrange(max(1, len(sig)-minoffset))
            nend = randrange(nstart+minoffset,
                             min(nstart+maxoffset, len(sig)+1))
            sample = Audio(sig[nstart:nend], sr)
        else:
            pp = os.path.join(self.root, self.filepaths[idx])
            sample = Audio(*randread(pp, self.minlen, self.maxlen, self.unit))

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
