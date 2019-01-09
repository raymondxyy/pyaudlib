"""Abstract dataset class.

This is a direct copy of PyTorch's dataset class:
https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#Dataset
with some omissions and additions.
"""

import bisect


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

    @property
    def all_files(self):
        """Hold all paths pointing to existing audio files."""
        raise NotImplementedError

    def __getitem__(self, index):
        """Return one sample at current index.

        For a generic dataset, a sample should have at least:
            'sr': sampling rate in int
            'data': audio waveform in numpy.ndarray
        """
        raise NotImplementedError

    def __len__(self):
        """Give the number of files available to be processed."""
        return len(self.all_files)

    def __add__(self, other):
        """Merge two Datasets."""
        return ConcatDataset([self, other])


class ASRDataset(Dataset):
    """An abstract dataset class for automatic speech recognition.

    All speech recognition datasets should subclass it.

    See Also
    --------
    audlib.data.wsj.ASRWSJ0, audlib.data.wsj.ASRWSJ1

    """

    @property
    def valid_files(self):
        """Hold all valid files to be processed.

        This property is a subset of `all_files` whose transcript:
            - is available in `self.transcripts`
            - contains only transcribable words
        """
        raise NotImplementedError

    @property
    def transcripts(self):
        """Obtain all transcripts for valid files."""
        raise NotImplementedError

    def __len__(self):
        """Return number of valid files in the dataset."""
        return len(self.valid_files)

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

    @property
    def valid_files(self):
        """Hold all valid files to be processed.

        A valid degraded file must have a reference clean file.
        """
        raise NotImplementedError

    def __len__(self):
        """Return number of valid files in the dataset."""
        return len(self.valid_files)

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

    @property
    def valid_files(self):
        """Hold all valid files to be processed.

        A valid file must have timestamps of speech-active regions.
        """
        raise NotImplementedError

    def __len__(self):
        """Return number of valid files in the dataset."""
        return len(self.valid_files)

    def __getitem__(self, index):
        """Return one sample at current index.

        For a SAD dataset, a sample should at least have:
            'sr': sampling rate
            'data': audio waveform
            'active': iterable of speech-active timestamps
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
