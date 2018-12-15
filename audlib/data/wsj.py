"""The Wall Street Journal Speech Recognition Datasets.

Contain both the pilot wsj0 and full wsj1.
"""
import glob
import os

from .dataset import Dataset, ASRDataset
from ..io.audio import audioread


def dot2tdict(dotpath):
    """Convert a .dot file to a dictionary of transcriptions.

    Parameters
    ----------
    dotpath: str
        Full path to a .dot transcription file.

    Returns
    -------
    tdict: dict of str
        tdict[condition][speaker ID][utterance ID] = transcription

    """
    tdict = {}
    with open(dotpath) as fp:
        for line in fp.readlines():
            line = line.strip().split()
            # Template
            # <transcription> <(utterance id)>
            trans, uid = ' '.join(line[:-1]), line[-1][1:-1]
            tdict[uid] = trans.upper()
    return tdict


def idx2flist(idxpath, root, train=True):
    """Convert a WSJ-style .idx file to a list of data paths.

    Parameters
    ----------
    idxpath: str
        Full path to an index file (.ndx).
    root: str
        Root directory to WSJ dataset.
    train: bool
        True if for training partition; else for test.

    Returns
    -------
    out: list of str
        A list of strings pointing to valid audio files.

    """
    def fix_path(path, wsj0=True):
        """Fix the inconsistencies between files index and path."""
        # 11_3_1:wsj0/sd_tr_s/001/001c0l01.wv1 ==>
        # 11-3.1/wsj0/sd_tr_s/001/001c0l01.wv
        return path.replace(' ', '').replace('_', '-', 1).replace(
                '_', '.', 1).replace(':', '/', 1)

    out = []
    if train:
        with open(idxpath) as fp:
            for line in fp.readlines():
                if line.startswith(';'):  # skip comment lines
                    continue
                fpath = fix_path(line.strip())
                if os.path.exists(os.path.join(root, fpath)):
                    out.append(fpath)
    else:
        with open(idxpath) as fp:
            for line in fp.readlines():
                if line.startswith(';'):  # skip comment lines
                    continue
                # these are all the inconsistencies between files
                # that index names like
                # 11_15_1:wsj0/sd_et_05/001/001o0v0g
                # and actual file path like
                # 11-15.1:wsj0/sd_et_05/001/001o0v0g.wv1 (or .wv2)
                # here we only pick wv1 unless only wv2's available
                fpath = fix_path(line.strip())
                if os.path.exists(os.path.join(root, fpath)):
                    out.append(fpath)
                elif os.path.exists(os.path.join(root, fpath+'.wv1')):
                    out.append(fpath+'.wv1')
                elif os.path.exists(os.path.join(root, fpath+'.wv2')):
                    out.append(fpath+'.wv2')
    return out


class WSJ(Dataset):
    """Instantiate a dataset from the WSJ0 dataset.

    Parameters
    ----------
    root: str
        The root directory of WSJ0.
    train: bool; default to True
        Instantiate the training partition if True; otherwise the test.
    filt: callable(str) -> bool
        A function that returns a boolean given a path to an audio. Use
        this to define training subsets with various conditions, or simply
        filter audio on length or other criteria.
    transform: callable(dict) -> dict
        User-defined transformation function.

    Returns
    -------
    A class wsj0 that has the following properties:
        - len(wsj0) == number of usable audio samples
        - wsj0[idx] == a dict that has the following structure
        sample: {
            'sr': sampling rate in int
            'data': audio waveform (of transform) in numpy.ndarray
        }

    """

    def __init__(self, root, train=True, filt=None, transform=None):
        """Instantiate a generic WSJ0 dataset by index files."""
        super(WSJ, self).__init__()
        self.root = root
        self.train = train
        self.filt = filt
        self.transform = transform
        self.tiregex = NotImplementedError
        self.eiregex = NotImplementedError

    @property
    def flist(self):
        """Build valid file list."""
        out = []  # holds all valid data paths
        if self.train:
            for idxpath in glob.iglob(self.tiregex):
                out.extend(idx2flist(idxpath, self.root))
        else:
            for idxpath in glob.iglob(self.eiregex):
                out.extend(idx2flist(idxpath, self.root, train=False))
        return list(filter(self.filt, out))

    def __str__(self):
        """Print out a summary of instantiated WSJ0."""
        report = """
            +++++ Summary for [{}][{} partition] +++++
            Total [{}] valid files to be processed.
        """.format(self.__class__.__name__, 'Train' if self.train else 'Test',
                   len(self.flist))
        return report

    def __repr__(self):
        """Representation of WSJ0."""
        return r"""{}({}, train={}, filt={}, transform={})
        """.format(self.__class__.__name__, self.root, self.train, self.filt,
                   self.transform)

    def __len__(self):
        """Return number of audio files to be processed."""
        return len(self.flist)

    def __getitem__(self, idx):
        """Retrieve the i-th example from the dataset."""
        fpath = os.path.join(self.root, self.flist[idx])

        data, sr = audioread(fpath)
        sample = {
            'sr': sr,
            'data': data
            }

        if self.transform:
            sample = self.transform(sample)

        return sample


class WSJ0(WSJ):
    """Instantiate WSJ0."""

    def __init__(self, root, train=True, filt=None, transform=None):
        super(WSJ0, self).__init__(root, train, filt, transform)
        # Validate directories of file indices and transcriptions
        self.tiregex = os.path.join(
            root, "11-13.1/wsj0/doc/indices/train/*.ndx")
        self.eiregex = os.path.join(
            root, "11-13.1/wsj0/doc/indices/test/*/*.ndx")


class WSJ1(WSJ):
    """Instantiate WSJ1."""

    def __init__(self, root, train=True, filt=None, transform=None):
        super(WSJ1, self).__init__(root, train, filt, transform)
        # Validate directories of file indices and transcriptions
        self.tiregex = os.path.join(
            root, "13-34.1/wsj1/doc/indices/si_tr_*.ndx")
        self.eiregex = os.path.join(root, "13-34.1/wsj1/doc/indices/h1_p0.ndx")


class ASRWSJ(ASRDataset):
    """docstring for WSJ0."""

    def __init__(self, dataset, transmap):
        """Instantiate a dataset from the WSJ0 dataset.

        Parameters
        ----------
        root: str
            The root directory of WSJ0.
        transmap: class
            A transcript map instance as defined in `audlib.asr.util`.
        train: bool; default to True
            Instantiate the training partition if True; otherwise the test.
        filt: callable(str) -> bool
            A function that returns a boolean given a path to an audio. Use
            this to define training subsets with various conditions, or simply
            filter audio on length or other criteria.
        transform: callable(dict) -> dict
            User-defined transformation function.

        Returns
        -------
        A class wsj0 that has the following properties:
            - len(wsj0) == number of usable audio samples
            - wsj0[idx] == a dict that has the following structure
            sample: {
                'sr': sampling rate in int
                'data': audio waveform (of transform) in numpy.ndarray
                'trans': transcription in str
                'label': label sequence in array of int
            }

        See Also
        --------
        audlib.asr.util.TranscriptMap

        """
        super(ASRWSJ, self).__init__(dataset, transmap)
        self.root = dataset.root
        self.dataset = dataset
        self.tmap = transmap

        # Validate directories of file transcriptions
        self.tdregex = NotImplementedError
        self.edregex = NotImplementedError

    @property
    def tdict(self):
        """Build transcript dictionary."""
        # Store all transcriptions in a dictionary
        # From a file path, retrieve its transcript with
        # self.transdict[cond][sid][uid]
        out = {}  # holds all valid transcriptions
        if self.dataset.train:
            for dotpath in glob.iglob(self.tdregex):
                cond, sid = dotpath.split('/')[-3:-1]
                if cond not in out:
                    out[cond] = {}
                if sid not in out[cond]:
                    out[cond][sid] = {}
                out[cond][sid].update(dot2tdict(dotpath))
        else:
            for dotpath in glob.iglob(self.edregex):
                cond, sid = dotpath.split('/')[-3:-1]
                if cond not in out:
                    out[cond] = {}
                if sid not in out[cond]:
                    out[cond][sid] = {}
                out[cond][sid].update(dot2tdict(dotpath))
        return out

    @property
    def vlist(self):
        """Prepare valid file list."""
        out = []  # holds valid file path indices
        self.oovs = {}  # holds out-of-vocab words
        for ii, fpath in enumerate(self.dataset.flist):
            fpath = os.path.join(self.root, fpath)
            if os.path.exists(fpath):
                cond, sid, uid = fpath.split('/')[-3:]
                uid = uid.split('.')[0]
                try:
                    trans = self.tdict[cond][sid][uid]
                except KeyError:  # no transcript
                    continue
                else:
                    if self.tmap.transcribable(trans):
                        out.append(ii)
                    else:
                        oov = self.tmap.trans2oov(trans)
                        for w in oov:
                            if w in self.oovs:
                                self.oovs[w] += oov[w]
                            else:
                                self.oovs[w] = oov[w]
        return out

    def __str__(self):
        """Print out a summary of instantiated WSJ0."""
        report = """
            +++++ Summary for [{}][{} partition] +++++
            Total [{}] files available.
            Total [{}] valid files to be processed (== len(self)).
            Total [{}] out-of-vocabulary words
            \t Some examples: [{}]
        """.format(self.__class__.__name__,
                   'Train' if self.dataset.train else 'Test',
                   len(self.dataset.flist), len(self.vlist), len(self.oovs),
                   ", ".join([e for e in self.oovs][:min(5, len(self.oovs))]))
        return report

    def __repr__(self):
        """Representation of WSJ0."""
        return r"""{}({}, {}, train={}, filt={}, transform={})
        """.format(self.__class__.__name__, self.root, self.tmap,
                   self.dataset.train, self.dataset.filt,
                   self.dataset.transform)

    def __len__(self):
        """Return number of audio files to be processed."""
        return len(self.vlist)

    def __getitem__(self, idx):
        """Retrieve the i-th example from the dataset."""
        fpath = os.path.join(self.root, self.dataset.flist[self.vlist[idx]])

        # Find corresponding transcript
        cond, sid, uid = fpath.split('/')[-3:]
        uid = uid.split('.')[0]
        trans = self.tdict[cond][sid][uid]

        # Convert transcript to label sequence
        label = self.tmap.trans2label(trans)

        data, sr = audioread(fpath)
        sample = {
            'sr': sr,
            'data': data,
            'trans': trans,
            'label': label
            }

        if self.dataset.transform:
            sample = self.dataset.transform(sample)

        return sample


class ASRWSJ0(ASRWSJ):
    """ASR dataset for WSJ0."""

    def __init__(self, dataset, transmap):
        super(ASRWSJ0, self).__init__(dataset, transmap)
        self.tdregex = os.path.join(
            self.root, "11-4.1/wsj0/transcrp/dots/*/*/*.dot")
        self.edregex = os.path.join(self.root, "11-14.1/wsj0/*/*/*.dot")


class ASRWSJ1(ASRWSJ):
    """ASR dataset for WSJ1."""

    def __init__(self, dataset, transmap):
        super(ASRWSJ1, self).__init__(dataset, transmap)
        self.tdregex = os.path.join(
            self.root, "13-34.1/wsj1/trans/wsj1/si_tr_*/*/*.dot")
        self.edregex = os.path.join(
            self.root, "13-34.1/wsj1/trans/wsj1/si_dt_20/*/*.dot")
