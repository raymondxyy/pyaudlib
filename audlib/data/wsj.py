"""The Wall Street Journal Datasets.

This module contains both the pilot wsj0 and full wsj1 datasets.
You MUST have the data on disk in order to use it.

Supported dataset formats:
    WSJ0 - generic WSJ0; offering audio samples
    ASRWSJ0 - WSJ0 for automatic speech recognition;
              offering audio samples, transcripts, labels
    WSJ1 - generic WSJ1; offering audio samples
    ASRWSJ1 - WSJ1 for automatic speech recognition;
              offering audio samples, transcripts, labels
"""
import glob
import os

from .dataset import Dataset, ASRDataset
from .datatype import Audio, SpeechTranscript
from ..io.audio import audioread


def dot2transcripts(dotpath):
    """Convert a .dot file to a dictionary of transcriptions.

    Parameters
    ----------
    dotpath: str
        Full path to a .dot transcription file.

    Returns
    -------
    transcripts: dict of str
        transcripts[condition][speaker ID][utterance ID] = transcript

    """
    transcripts = {}
    with open(dotpath) as fp:
        for line in fp.readlines():
            line = line.strip().split()
            # Template
            # <transcription> <(utterance id)>
            trans, uid = ' '.join(line[:-1]), line[-1][1:-1]
            transcripts[uid] = trans.upper()
    return transcripts


def idx2paths(idxpath, root, train=True):
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
    """Generic dataset framework for WSJ0 and WSJ1.

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
        - wsj0[idx] == an Audio object

    """

    def __init__(self, root, ndxpath, train=True, filt=None, transform=None):
        """Instantiate a generic WSJ0 dataset by index files."""
        super(WSJ, self).__init__()
        self.root = root
        self.train = train
        self.transform = transform
        ndxpath = os.path.join(root, ndxpath)

        self._filepaths = []  # holds all valid data paths
        if self.train:
            for idxpath in glob.iglob(ndxpath):
                self._filepaths.extend(idx2paths(idxpath, self.root))
        else:
            for idxpath in glob.iglob(ndxpath):
                self._filepaths.extend(
                    idx2paths(idxpath, self.root, train=False))
        self._filepaths = list(filter(filt, self._filepaths))

    @property
    def filepaths(self):
        """Build valid file list."""
        return self._filepaths

    def __str__(self):
        """Print out a summary of instantiated WSJ0."""
        report = """
            +++++ Summary for [{}][{} partition] +++++
            Total [{}] valid files to be processed.
        """.format(self.__class__.__name__,
                   'Train' if self.train else 'Test',
                   len(self.filepaths))

        return report

    def __repr__(self):
        """Representation of WSJ0."""
        return r"""{}({}, train={}, filt={}, transform={})
        """.format(self.__class__.__name__, self.root, self.train, self.filt,
                   self.transform)

    def __len__(self):
        """Return number of audio files to be processed."""
        return len(self.filepaths)

    def __getitem__(self, idx):
        """Get the idx-th example from the dataset."""
        sample = Audio(*audioread(os.path.join(self.root,
                                               self._filepaths[idx])))

        if self.transform:
            sample = self.transform(sample)

        return sample


class WSJ0(WSJ):
    """Generic WSJ0 framework."""

    def __init__(self, root, train=True, filt=None, transform=None):
        """Instantiate generic WSJ0."""
        if train:
            ndxpath = "11-13.1/wsj0/doc/indices/train/*.ndx"
        else:
            ndxpath = "11-13.1/wsj0/doc/indices/test/*/*.ndx"
        super(WSJ0, self).__init__(root, ndxpath, train, filt, transform)


class WSJ1(WSJ):
    """Generic WSJ1 framework."""

    def __init__(self, root, train=True, filt=None, transform=None):
        """Instantiate generic WSJ1."""
        if train:
            ndxpath = "13-34.1/wsj1/doc/indices/si_tr_*.ndx"
        else:
            ndxpath = "13-34.1/wsj1/doc/indices/h1_p0.ndx"
        super(WSJ1, self).__init__(root, ndxpath, train, filt, transform)


class ASRWSJ(ASRDataset):
    """ASR framework for WSJ0 and WSJ1."""

    def __init__(self, dataset, transmap, transpath, verbose=False):
        """Instantiate a WSJ dataset for speech recognition.

        Parameters
        ----------
        dataset: Dataset class
            Dataset to be processed.
        transmap: TranscriptMap class
            A transcript map instance as defined in `audlib.asr.util`.
        transpath: str
            Path pattern to transcripts.
        verbose: bool
            Print info while processing?

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
        # TODO: Reduce cyclomatic complexity
        super(ASRWSJ, self).__init__()
        self.dataset = dataset
        self.transmap = transmap
        self.verbose = verbose
        # Store all transcriptions in a dictionary
        # From a file path, retrieve its transcript with
        # self.transdict[cond][sid][uid]
        if self.verbose:
            print("Preparing transcript tree.")
        self._transcripts = {}  # holds all valid transcriptions
        for dotpath in glob.iglob(transpath):
            cond, sid = dotpath.split('/')[-3:-1]
            if cond not in self._transcripts:
                self._transcripts[cond] = {}
            if sid not in self._transcripts[cond]:
                self._transcripts[cond][sid] = {}
            self._transcripts[cond][sid].update(dot2transcripts(dotpath))

        if self.verbose:
            print("Preparing valid files.")
        self._filepaths = []
        self.oovs = {}  # holds out-of-vocab words
        self.vocab_hist = [0] * len(self.transmap)
        for ii, fpath in enumerate(self.dataset.filepaths):
            if not ((ii+1) % 1000) and self.verbose:
                print("Processing [{}/{}] files.".format(
                    ii+1, len(self.dataset)))
            fpath = os.path.join(self.dataset.root, fpath)
            if os.path.exists(fpath):
                cond, sid, uid = fpath.split('/')[-3:]
                uid = uid.split('.')[0]
                try:
                    trans = self.transcripts[cond][sid][uid]
                except KeyError:  # no transcript
                    continue
                else:
                    if "[BAD_RECORDING]" in trans:
                        continue
                    if self.transmap.transcribable(trans):
                        self._filepaths.append(ii)
                        # Accumulate vocab stats here
                        for ll in self.transmap.trans2label(trans):
                            self.vocab_hist[ll] += 1
                    else:
                        if verbose:
                            print("OOV in [{}]: [{}]".format(fpath, trans))
                        oov = self.transmap.trans2oov(trans)
                        for w in oov:
                            if w in self.oovs:
                                self.oovs[w] += oov[w]
                            else:
                                self.oovs[w] = oov[w]

    @property
    def filepaths(self):
        """Return valid file index list."""
        return self._filepaths

    @property
    def transcripts(self):
        """Return trancript dictionary for valid files."""
        return self._transcripts

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
                   len(self.dataset.filepaths), len(
                       self.filepaths), len(self.oovs),
                   ", ".join([e for e in self.oovs][:min(5, len(self.oovs))]))
        return report

    def __repr__(self):
        """Representation of WSJ0."""
        return r"""{}({}, {})
        """.format(self.__class__.__name__, self.dataset, self.transmap)

    def __len__(self):
        """Return number of audio files to be processed."""
        return len(self.filepaths)

    def __getitem__(self, idx):
        """Retrieve the i-th example from the dataset."""
        fpath = os.path.join(
            self.dataset.root, self.dataset.filepaths[self.filepaths[idx]])

        # Find corresponding transcript
        cond, sid, uid = fpath.split('/')[-3:]
        uid = uid.split('.')[0]
        trans = self.transcripts[cond][sid][uid]

        # Convert transcript to label sequence
        label = self.transmap.trans2label(trans)

        sample = SpeechTranscript(*audioread(fpath),
                                  transcript=trans, label=label)

        if self.dataset.transform:
            sample = self.dataset.transform(sample)

        return sample


class ASRWSJ0(ASRWSJ):
    """ASR dataset for WSJ0."""

    def __init__(self, dataset, transmap, verbose=False):
        """Instantiate WSJ0 for automatic speech recognition."""
        if dataset.train:
            transpath = os.path.join(
                dataset.root, "11-4.1/wsj0/transcrp/dots/*/*/*.dot")
        else:
            transpath = os.path.join(dataset.root, "11-14.1/wsj0/*/*/*.dot")
        super(ASRWSJ0, self).__init__(dataset, transmap, transpath, verbose)


class ASRWSJ1(ASRWSJ):
    """ASR dataset for WSJ1."""

    def __init__(self, dataset, transmap, verbose=False):
        """Instantiate WSJ1 for automatic speech recognition."""
        if dataset.train:
            transpath = os.path.join(
                dataset.root, "13-34.1/wsj1/trans/wsj1/si_tr_*/*/*.dot")
        else:
            transpath = os.path.join(
                dataset.root, "13-34.1/wsj1/trans/wsj1/si_dt_20/*/*.dot")
        super(ASRWSJ1, self).__init__(dataset, transmap, transpath, verbose)
