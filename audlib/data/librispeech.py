# coding: utf-8

"""Dataset classes for LibriSpeech."""
import os
import random
import numpy as np

from .dataset import SIDDataset, audioread, lsfiles, LstDataset
from .datatype import AudioSpeaker

from ..io.audio import audioinfo


class LibriSpeakers(SIDDataset):
    """Use LibriSpeech for speaker identification.

    The dataset should follow this directory pattern:
    .
    ├── BOOKS.TXT
    ├── CHAPTERS.TXT
    ├── dev-clean
    ├── dev-other (optional)
    ├── LICENSE.TXT
    ├── README.TXT
    ├── SPEAKERS.TXT
    ├── test-clean
    ├── test-other (optional)
    ├── train-clean-100
    ├── train-clean-360 (optional)
    └── train-other-500 (optional)

    See http://www.openslr.org/12 for more information.

    """

    @staticmethod
    def isflac(path):
        """Original LibriSpeech has .flac files."""
        return path.endswith('.flac')

    @staticmethod
    def split(durations, proportions):
        """Split some durations into several partitions based on proportions.

        Parameters
        ----------
        durations: numpy.ndarray
            A list of non-negative durations.
        proportions: tuple(float)
            Proportions summing up to 1.

        Returns
        -------
        idx: tuple(int)
            List indices for durations.

        """
        dur_accum = np.cumsum(durations)
        dur_accum = dur_accum / dur_accum[-1]
        return (np.searchsorted(
            dur_accum, p) for p in np.cumsum(proportions)[:-1])

    def __init__(self, root, splits=(1,), filt=None, read=None,
                 transform=None, shuffle=False, seed=0):
        """Instantiate an audio dataset.

        Parameters
        ----------
        root: str
            Root directory of a dataset.
        split: tuple of float, ()
            Train, development, and test split.
            Examples:
                (1,) (default): no split
                (.8, .2): 80% train, 20% test per speaker
                (.8, .1, .1): 80% train, 10% developement, 10% test per speaker
                (.2,): WRONG format; must sum to 1
                (.1,.2,.3,.4): WRONG format; must not exceed 3 partitions
        filt: callable, None
            Filter function to be applied on each file path.
            Default accepts .flac only.
        read: callable, None
            Function to be called on each file path to get the signal.
            Default to `audioread`.
        transform: callable, None
            Transform to be applied on each sample after read in.

        See Also
        --------
        datatype.Audio

        """
        super(LibriSpeakers).__init__()
        assert 1 <= len(splits) <= 3 and sum(splits) == 1, "Invalid split!"

        self.root = root
        self._filepaths = lsfiles(
            root,
            filt=(lambda p: self.isflac(p) and filt(p)) if filt else self.isflac,
            relpath=True
        )

        if shuffle:
            random.seed(seed)
            random.shuffle(self._filepaths)

        # Speaker-duration dry run
        # spkr_dur[sid] = [(idx, duration)]
        self._spkr_dur = {}
        for ii, path in enumerate(self._filepaths):
            if (ii+1) % 1000 == 0:
                print(f"Accumulating [{ii+1}/{len(self._filepaths)}] stats.")
            _, sid, *_ = path.split('/')
            path = os.path.join(self.root, path)
            if sid in self._spkr_dur:
                self._spkr_dur[sid].append((ii, audioinfo(path).duration))
            else:
                self._spkr_dur[sid] = [(ii, audioinfo(path).duration)]

        self._spkr_label = {k: i for i, k in enumerate(self._spkr_dur)}

        # Now split datasets if needed
        if len(splits) < 2:  # no split
            print("Attribute all data to a single trainset.")
            self.trainset = LstDataset(self._filepaths, self.root,
                                       read=self.read, transform=transform)
            self.validset = self.testset = None
        elif len(splits) == 2:  # train-test split
            print(f"Split data into a train and test set.")
            trainpaths, testpaths = [], []
            for _, idx_dur in self._spkr_dur.items():
                idx, dur = zip(*idx_dur)
                ii, = self.split(dur, splits)
                trainpaths.extend(idx[:ii])
                testpaths.extend(idx[ii:])
            self.trainset = LstDataset(
                [self._filepaths[i] for i in trainpaths], self.root,
                read=self.read, transform=transform)
            self.testset = LstDataset(
                [self._filepaths[i] for i in testpaths], self.root,
                read=self.read, transform=transform)
            self.validset = None
        else:  # train-valid-test split
            print(f"Split data into a train, valid, and test set.")
            trainpaths, validpaths, testpaths = [], [], []
            for _, idx_dur in self._spkr_dur.items():
                idx, dur = zip(*idx_dur)
                ii, jj = self.split(dur, splits)
                trainpaths.extend(idx[:ii])
                validpaths.extend(idx[ii:jj])
                testpaths.extend(idx[jj:])
            self.trainset = LstDataset(
                [self._filepaths[i] for i in trainpaths], self.root,
                read=self.read, transform=transform)
            self.validset = LstDataset(
                [self._filepaths[i] for i in validpaths], self.root,
                read=self.read, transform=transform)
            self.testset = LstDataset(
                [self._filepaths[i] for i in testpaths], self.root,
                read=self.read, transform=transform)

        self.customread = read

    @property
    def filepaths(self):
        """Return all valid file paths in a list."""
        return self._filepaths

    def spkr2label(self, spkr):
        """Convert a speaker identifier to a unique integer."""
        return self._spkr_label[spkr]

    def read(self, path):
        """Read a single file from path."""
        sid = self._spkr_label[os.path.basename(path).split('-')[0]]
        if self.customread:
            return AudioSpeaker(*self.customread(path), speaker=sid)
        return AudioSpeaker(*audioread(path), sid)

    def __len__(self):
        """Return number of valid audio files."""
        return len(self._filepaths)

    def __getitem__(self, idx):
        """Get i-th valid item after reading in and transform."""
        print("Indexing a LibriSpeech class does nothing!")

    def __str__(self):
        """Print out a summary of instantiated dataset."""
        report = """
            +++++ Summary for [{}] +++++
            Total [{}] valid files to be processed.\n
            [Training]: \t{}
            [Validation]: \t{}
            [Test]: \t{}
        """.format(self.__class__.__name__, len(self._filepaths),
                   self.trainset, self.validset, self.testset)

        return report
