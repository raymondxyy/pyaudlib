# coding: utf-8

"""Dataset derived from the ESC-50 dataset for audio event classification."""
import os

from .dataset import AudioDataset, audioread
from .datatype import Audio


class ESCAudio(Audio):
    """A data structure for ESC-50 audio."""
    __slots__ = 'target', 'category'

    def __init__(self, signal=None, samplerate=None, target=None,
                 category=None):
        super(ESCAudio, self).__init__(signal, samplerate)
        self.target = target
        self.category = category


class ESC50(AudioDataset):
    """Generic ESC-50 dataset for audio event classification.

    The dataset should have the following directory structure:
    .
    ├── audio
    ├── esc50.gif
    ├── LICENSE
    ├── meta
    ├── pytest.ini
    ├── README.md
    ├── requirements.txt
    └── tests

    See more information on https://github.com/karoldvl/ESC-50.
    """
    @staticmethod
    def metaread(path):
        """Read the esc50.csv meta file to a dictionary.

        File format is:
            filename,fold,target,category,esc10,src_file,take
            1-100032-A-0.wav,1,0,dog,True,100032,A
            ...
        """
        meta = {}
        with open(path) as fp:
            next(fp)
            for line in fp:
                fn, fd, tar, cat, esc10, src, take = line.rstrip().split(',')
                meta[fn] = (int(fd), int(tar), cat, bool(esc10), src, take)
        return meta

    @staticmethod
    def lstallcategories(meta):
        """List all categories in a meta data file."""
        categories = {}
        for _, (_, tar, cat, *_) in meta.items():
            if cat not in categories:
                categories[cat] = tar

        return categories

    def __init__(self, root, categories=None, filt=None, transform=None):
        """Instantiate a ESC-50 dataset.

        Parameters
        ----------
        root: str
            Full path to root directory.
        categories: list of str, None
            Categories to take. Default takes all 50 categories.
        filt: callable, optional
            Filters to be applied on each audio path. Default to None.
        transform: callable(ESCAudio) -> ESCAudio
            User-defined transformation function.

        See Also
        --------
        ESCAudio
        """
        self.meta = self.metaread(os.path.join(root, 'meta/esc50.csv'))
        self._all_categories = self.lstallcategories(self.meta)
        if categories:
            assert all(c in self._all_categories for c in categories)
            self.categories = {c: self._all_categories[c] for c in categories}
        else:
            self.categories = self._all_categories

        def _filt(path):
            if not path.endswith('.wav'):
                return False
            bname = os.path.basename(path)
            if filt:
                return filt(path) and (self.meta[bname][2] in self.categories)
            else:
                return self.meta[bname][2] in self.categories

        super(ESC50, self).__init__(root, filt=_filt, read=self.read,
                                    transform=transform)

    def read(self, path):
        """Parse a path into fields, and read audio."""
        tar, cat = self.meta[os.path.basename(path)][1:3]
        sig, ssr = audioread(os.path.join(self.root, path))

        return ESCAudio(sig, ssr, tar, cat)

    def __repr__(self):
        """Representation of ESC-50."""
        return r"""{}({}, categories={})
        """.format(self.__class__.__name__, self.root, self.categories)

    def __str__(self):
        """Print out a summary of instantiated dataset."""
        catcnts = {c: 0 for c in self.categories}
        for fn in self._filepaths:
            catcnts[self.meta[os.path.basename(fn)][2]] += 1
        report = """
            +++++ Summary for [{}] +++++
            Total [{}] valid files to be processed.
            Total [{}/50] categories appear in this set.
            [Category]: [counts]\n{}
        """.format(self.__class__.__name__,
                   len(self._filepaths),
                   len(self.categories),
                   "\n".join(
                       f"\t\t[{cat:>16}]: [{cn:>2}]"
                       for cat, cn in catcnts.items()))

        return report
