"""Abstract dataset class.

This is a direct copy of PyTorch's dataset class:
https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#Dataset
with some omissions and additions.
"""
import math
import os
import bisect

from ..io.batch import lsfiles
from ..io.audio import audioread, audioinfo
from .datatype import Audio


class Dataset(object):
    """An abstract class representing a generic dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.

    Dataset should be used if only the the audio waveform data is of interest.
    Otherwise, a more specific dataset (see below) is likely to be more useful.

    In other words, use Dataset for:
        - Batch processing (e.g. extract feature, accumulate statistics)
        - Multi-class classification tasks that are easy to obtain labels
        - Other tasks that don't belong to more specific classes below

    See also:
        - ``ASRDataset`` for automatic speech recognition
        - ``SEDataset`` for speech enhancement
        - ``SADDataset`` for speech activity detection

    """

    def __getitem__(self, index):
        """Return one sample at current index.

        For a generic dataset, a sample should have at least:
            'sr': sampling rate in int
            'data': audio waveform in numpy.ndarray
        """
        raise NotImplementedError

    def __len__(self):
        """Give the number of files available to be processed."""
        raise NotImplementedError

    def __add__(self, other):
        """Merge two Datasets."""
        return ConcatDataset([self, other])


class AudioDataset(Dataset):
    """A dataset that gets all audio files from a directory."""

    @staticmethod
    def isaudio(path):
        return path.endswith(('.wav', '.flac', '.sph'))

    @staticmethod
    def read(path):
        """Read audio and put in an Audio object."""
        return Audio(*audioread(path))

    def __init__(self, root, filt=None, read=None, transform=None):
        """Instantiate an audio dataset.

        Parameters
        ----------
        root: str
            Root directory of a dataset.
        sr: int
            Sampling rate in Hz.
        read: callable, optional
            Function to be called on each file path to get the signal.
            Default to `audioread`.
        filt: callable, optional
            Filter function to be applied on each file path.
            Default to `isaudio`, which accepts every file ended in .wav, .sph,
            or .flac.
        transform: callable, optional
            Transform to be applied on each sample after read in.
            Default to None.

        See Also
        --------
        datatype.Audio

        """
        super(AudioDataset).__init__()
        self.root = root
        self._filepaths = lsfiles(root, filt=filt if filt else self.isaudio,
                                  relpath=True)
        self.customread = read
        self.transform = transform

    @property
    def filepaths(self):
        """Return all valid file paths in a list."""
        return self._filepaths

    def __len__(self):
        """Return number of valid audio files."""
        return len(self._filepaths)

    def __getitem__(self, idx):
        """Get i-th valid item after reading in and transform."""
        if self.customread:
            sample = self.customread(
                os.path.join(self.root, self._filepaths[idx]))
        else:
            sample = self.read(
                os.path.join(self.root, self._filepaths[idx]))

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __str__(self):
        """Print out a summary of instantiated dataset."""
        report = """
            +++++ Summary for [{}] +++++
            Total [{}] valid files to be processed.
        """.format(self.__class__.__name__, len(self._filepaths))

        return report


class LongFile(Dataset):
    """Treat a long audio file as a dataset and sample fixed-length segments."""
    def __init__(self, path, seglength, segshift, transform=None):
        """Instantiate a LongFile dataset."""
        assert os.path.exists(path), "File does not exist!"
        self.path = path
        self.info = audioinfo(self.path)
        sr = self.info.samplerate
        self.segshift = int(segshift*sr)
        assert self.segshift > 1, "Insufficient segshift!"
        self.seglength = int(seglength*sr)

    def __getitem__(self, idx):
        idx %= len(self)
        ns = idx*self.segshift
        return Audio(*audioread(self.path, frames=self.seglength,
                                start=ns, fill_value=0))

    def __len__(self):
        return math.ceil(
            (self.info.frames - self.seglength) / self.segshift) + 1


class LstDataset(Dataset):
    """A dataset that gets all audio files from a list of file paths."""

    def __init__(self, lst, root=None, read=None, transform=None):
        """Instantiate an audio dataset.
False
        Parameters
        ----------
        lst: list(str) or str
            Each entry should point to a valid file.
            If a str, assume a file-listing file that has file spath per line.

        Keyword Parameters
        ------------------
        root: str, None
            Root directory of a dataset.
        sr: int
            Sampling rate in Hz.
        read: callable, optional
            Function to be called on each file path to get the signal.
            Default to `audioread`.
        transform: callable, optional
            Transform to be applied on each sample after read in.
            Default to None.

        See Also
        --------
        datatype.Audio

        """
        super(LstDataset).__init__()
        self.root = root
        if isinstance(lst, list):
            self._filepaths = lst
        elif isinstance(lst, str):
            self._filepaths = [line.rstrip('\n') for line in open(lst)]
        else:
            raise NotImplementedError
        self.customread = read
        self.transform = transform

    @property
    def filepaths(self):
        """Return all valid file paths in a list."""
        return self._filepaths

    def read(self, path):
        """Read a single file from path."""
        if self.customread:
            return self.customread(path)
        return Audio(*audioread(path))

    def __len__(self):
        """Return number of valid audio files."""
        return len(self._filepaths)

    def __getitem__(self, idx):
        """Get i-th valid item after reading in and transform."""
        path = os.path.join(self.root, self._filepaths[idx]) if self.root else\
            self._filepaths[idx]
        sample = self.read(path)
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __str__(self):
        """Print out a summary of instantiated dataset."""
        report = """
            +++++ Summary for [{}] +++++
            Total [{}] valid files to be processed.
        """.format(self.__class__.__name__, len(self._filepaths))

        return report


class ASRDataset(Dataset):
    """An abstract dataset class for automatic speech recognition.

    All speech recognition datasets should subclass it.

    See Also
    --------
    audlib.data.wsj.ASRWSJ0, audlib.data.wsj.ASRWSJ1

    """

    def __getitem__(self, index):
        """Return one sample at current index.

        For a speech recognition dataset, a sample should at least have:
            'sr': sampling rate in int
            'data': audio waveform in numpy.ndarray
            'trans': transcript in str
            'label': label sequence in array of int
        """
        raise NotImplementedError


class SEDataset(Dataset):
    """An abstract dataset class for Speech Enhancement.

    All speech enhancement datasets should subclass it.

    See Also
    --------
    audlib.data.rats.SERATS_SAD, audlib.data.vctk.SEVCTK2chan

    """

    def __getitem__(self, index):
        """Return one sample at current index.

        For a speech enhancement dataset, a sample should at least have:
            'chan1': {'sr': sampling rate, 'data': degraded audio waveform}
            'clean': {'sr': sampling rate, 'data': reference audio waveform}
        """
        raise NotImplementedError


class SADDataset(Dataset):
    """An abstract dataset class for Speech Activity Detection.

    All SAD datasets should subclass it.

    See Also
    --------
    audlib.data.rats.RATS_SAD

    """

    def __getitem__(self, index):
        """Return one sample at current index.

        For a SAD dataset, a sample should at least have:
            'sr': sampling rate
            'data': audio waveform
            'active': iterable of speech-active timestamps
        """
        raise NotImplementedError


class SIDDataset(Dataset):
    """An abstract dataset class for Speaker IDentification.

    All SID datasets should subclass it.

    See Also
    --------
    audlib.data.librispeech.LibriSpeakers

    """

    def spkr2label(self, spkr):
        """Convert a unique speaker identifier to a unique integer label."""
        raise NotImplementedError

    def __getitem__(self, index):
        """Return one sample at current index.

        For a SAD dataset, a sample should at least have:
            Audio: datatype.Audio
            speaker: str, int, or any unique identifier
        """
        raise NotImplementedError


class ConcatDataset(Dataset):
    """Dataset to concatenate multiple datasets.

    It is useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.
    """

    @staticmethod
    def cumsum(sequence):
        """Cumulative sum."""
        r, s = [], 0
        for e in sequence:
            ll = len(e)
            r.append(ll + s)
            s += ll
        return r

    def __init__(self, datasets):
        """Instantiate a concatenated dataset from a list of datasets.

        Parameters
        ----------
        datasets: iterable of datasets
            List of datasets to be concatenated

        """
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        """Total number of files in combined datasets."""
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        """Obtain idx-th sample."""
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    def __str__(self):
        """Print each dataset's string."""
        return "\n".join(str(dd) for dd in self.datasets)


class Subset(Dataset):
    """Subset of a dataset at specified indices."""

    def __init__(self, dataset, indices):
        """Instantiate a subset from a full dataset and a list of indices.

        Parameters
        ----------
        dataset: Dataset
            The whole Dataset
        indices: iterables of int
            Indices in the whole set selected for subset

        """
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        """Get an item from the subset."""
        return self.dataset[self.indices[idx]]

    def __len__(self):
        """Return number of files in the subset."""
        return len(self.indices)
