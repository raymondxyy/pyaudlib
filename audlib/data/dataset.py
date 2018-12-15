"""Abstract dataset class.

This is a direct copy of PyTorch's dataset class:
https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#Dataset

with some omissions and additions.
"""

import bisect
import os
import warnings


class Dataset(object):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    @property
    def flist(self):
        """Hold a list of absolute File paths.

        Alternatively, each member could be elements that allow the absolute
        file path to be composed. Each path should exist and point to a valid
        audio file.
        """
        raise NotImplementedError

    def __getitem__(self, index):
        """Return one sample at current index.

        For a generic dataset, a sample should be defined as:
        {
            'sr': sampling rate in int
            'data': audio waveform in numpy.ndarray
        }
        """
        raise NotImplementedError

    def __len__(self):
        """Give the number of files available to be processed."""
        return len(self.flist)

    def __add__(self, other):
        """Merge two datasets."""
        return ConcatDataset([self, other])

    @staticmethod
    def exists(path):
        """Assert the existence of a path at `path`."""
        assert os.path.exists(path), "[{}] does not exist!".format(path)


class ASRDataset(Dataset):
    """An abstract class for automatic speech recognition.

    All speech recognition datasets should subclass it.
    """

    @property
    def vlist(self):
        """Hold a LIST of Valid file paths.

        Each path should fulfill the following requirements:
            - exists in `self.flist`
            - has a transcript available in `self.tdict`
            - its transcript has transcribable words (i.e., label is available)
        """
        raise NotImplementedError

    @property
    def tdict(self):
        """Hold a DICTionary of Transcripts.

        The exact data structure of the dictionary could vary, but it should
        be able to get to the transcript provided an entry in `self.vlist`.
        """
        raise NotImplementedError

    def __init__(self, dataset, transmap):
        """Instantiate a ASR dataset.

        `dataset` should have the following properties:
            - `root` for root directory
            - `flist` for list of valid audio files
        """
        super(ASRDataset, self).__init__()
        self.root = dataset.root
        self.tmap = transmap

    def __len__(self):
        """Return number of valid files with valid transcript."""
        return len(self.vlist)

    def __getitem__(self, index):
        """Return one sample at current index.

        For a speech recognition dataset, a sample should be defined as:
        {
            'sr': sampling rate in int
            'data': audio waveform in numpy.ndarray
            'trans': transcript in str
            'label': label sequence in array of int
        }
        """
        raise NotImplementedError


class ConcatDataset(Dataset):
    """Dataset to concatenate multiple datasets.

    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes


class Subset(Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)
