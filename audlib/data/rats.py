# coding: utf-8

"""Define RATS datasets."""

import glob
import os

import soundfile as sf

from .dataset import SEDataset, Dataset
from .datatype import NoisySpeech, Audio
from ..io.audio import audioread, audioinfo
from ..io.batch import genfiles


class SpeechActivity(Audio):
    """A data structure for speech activity detection.

    speechtype can only be one of the following:
        - None --> Undecided
        - 's' --> spech
        - 'n' --> nonspeech
    """
    __slots__ = "speechtype"

    def __init__(self, signal=None, samplerate=None, speechtype=None):
        super(SpeechActivity, self).__init__(signal, samplerate)
        self.speechtype = speechtype


class SERATS_SAD(SEDataset):
    """RATS Speech Activity Detection dataset (LDC2015S02).

    NOTE: This class **requires alignment** of noisy speech to clean speech.
    All 'aligned directories' are not in the original RATS_SAD. They are
    obtained by running noisy-clean pair through Dan Ellis' alignment tool
    [skewview](https://labrosa.ee.columbia.edu/projects/skewview/).
    This class assumes such alignment has already been done. Please take a step
    back and do the alignment before using this class if it's not done. After
    alignement, put the aligned directories in the same level of dev-1, dev-2,
    and train. The directory structure should look like:

        data
        ├── aligned-dev-1
        │   ├── audio
        │   └── sad
        ├── aligned-dev-2
        │   ├── audio
        │   └── sad
        ├── dev-1
        │   ├── audio
        │   └── sad
        ├── dev-2
        |   ├── audio
        |   └── sad
        ├── train
        │   ├── audio
        │   └── sad
        └── aligned-train
            ├── audio
            └── sad
    """

    def __init__(self, root, subset='dev-1', channels='AH', select=None,
                 transform=None):
        """Create a dataset from `channels` and takes specified transform.

        Parameters
        ----------
        root: str
            Root directory of the RATS_SAD dataset.
        subset: str
            One of dev-1, dev-2, or train.
        channels: str
            A selection of RATS channels. Available channels are [A-H].
        transform: callable [None]
            Transform to be done on each audio sequence.

        Returns
        -------
        A dataset that yields a {noisy, clean, vad} feature dict at indexing.
        Note that although each index is viewed as one example, in fact
        multiple examples could be retrieved. Specifically, each example will
        provide:
            - noisy feature: [M x T x F]
            - clean feature: [M x T x F]
            - vad reference: [M x T]
        where M = min(maxegs, maximum number of slices of an audio sequence).
        Since M could vary, a custom `collate_fn` is needed to patch examples
        in this dataset into dataloader.

        """
        self.root = root
        self.subset = subset
        self.channels = channels
        if subset not in ('dev-1', 'dev-2', 'train'):
            raise ValueError("Error: Must be one of dev-1/dev-2/train")
        self.noisydir = os.path.join(root, 'data',
                                     'aligned-{}'.format(subset), 'audio')
        self.cleandir = os.path.join(root, 'data', subset, 'audio/src')
        self.vaddir = os.path.join(self.root, 'data', subset, 'sad/src')
        assert os.path.exists(self.noisydir), \
            "Noisy directory does not exist!".format(self.noisydir)
        assert os.path.exists(self.cleandir), \
            "Clean directory does not exist!".format(self.cleandir)
        assert os.path.exists(self.vaddir), \
            "VAD directory does not exist!".format(self.vaddir)

        self.select = select
        self.transform = transform

        self._filepaths = []
        for chan in self.channels:
            for noisy in glob.iglob(os.path.join(
                    self.noisydir, chan, '*.flac')):
                fid = os.path.basename(noisy)[:5]
                clean = glob.glob(os.path.join(self.cleandir,
                                               "{}*.flac".format(fid)))[0]
                vad = glob.glob(os.path.join(self.vaddir,
                                             "{}*.tab".format(fid)))[0]
                self._filepaths.append((noisy, clean, vad))

    @property
    def filepaths(self):
        """Get all valid files."""
        return self._filepaths

    def __len__(self):
        """Return number of valid file paths."""
        return len(self._filepaths)

    def __getitem__(self, idx):
        """Convert (noisy, clean, vad) paths to features on indexing."""
        noisy, clean, vad = self.filepaths[idx]
        if self.select is not None:
            # Quite a hacky way because noisy and clean have unequal lengths
            if sf.info(noisy).frames > sf.info(clean).frames:
                shorter = clean
            else:
                shorter = noisy
            nstart, nend = self.select(shorter)
        else:
            nstart, nend = 0, None
        sigx, sr1 = audioread(noisy, start=nstart, stop=nend)
        sigs, sr2 = audioread(clean, start=nstart, stop=nend)
        assert sr1 == sr2

        # Equalize lengths if necessary
        if len(sigx) > len(sigs):
            sigx = sigx[:len(sigs)]
        elif len(sigx) < len(sigs):
            sigs = sigs[:len(sigx)]

        # Calculate new vad timestamps
        offset = nstart*1. / sr1
        vadref = self.tabread(vad)
        vadref = [(ts-offset, te-offset) for ts, te in vadref]
        sample = NoisySpeech(noisy=Audio(sigx, sr1),
                             clean=Audio(sigs, sr2), vad=vadref)

        if self.transform:
            sample = self.transform(sample)

        return sample

    @staticmethod
    def tabread(tabpath):
        """Read .tab file to speech-only timestamps."""
        tstamps = []
        with open(tabpath, 'r') as fp:
            for line in fp:
                line = line.rstrip().split()
                stype, tstart, tend = line[4], float(line[2]), float(line[3])
                if stype == 'S':  # speech
                    tstamps.append((tstart, tend))
        return tstamps


class RATS_SAD(Dataset):
    """The orignal LDC2015S02 dataset for speech activity detection.

    The dataset directory should follow the structure below.

    """
    def __init__(self, root, partition, filt=None, transform=None):
        """Instantiate an ASVspoof dataset.

        Parameters
        ----------
        root: str
            The root directory of AVSpoof.
        partition: str
            One of 'train', 'dev-1', or 'dev-2'.
        filt: callable() --> bool
            Optional user-defined filter function.
        transform: callable(SpeechActivity) -> SpeechActivity
            User-defined transformation function.

        Returns
        -------
        An class instance `rats_sad` that has the following properties:
            - len(rats_sad) == number of usable audio samples
            - rats_sad[idx] == a SpeechActivity instance

        See Also
        --------
        SpeechActivity

        """
        if partition == 'train':
            root = os.path.join(root, 'data/train')
        elif partition == 'dev-1':
            root = os.path.join(root, 'data/dev-1')
        elif partition == 'dev-2':
            root = os.path.join(root, 'data/dev-2')
        else:
            raise ValueError("partition must be one of train/valid/test.")
        self.root = root
        self.audroot = os.path.join(root, 'audio')
        self.sadroot = os.path.join(root, 'sad')
        assert all(os.path.exists(dd) for dd in [self.audroot, self.sadroot])
        self.partition = partition

        # Read all segments from .tab files
        self._allsegs = []
        for tt in genfiles(self.sadroot,
                           filt=lambda p: p.endswith('.tab'), relpath=True):
            # check existence of audio first
            aa = tt.replace('.tab', '.flac')
            if not os.path.exists(os.path.join(self.audroot, aa)):
                print(f"{tt} does not exist. Skip...")
                continue
            segs = filter(filt, self.tab2segs(os.path.join(self.sadroot, tt)))
            self._allsegs.extend([(aa, seg) for seg in segs])

        self.transform = transform

    def read(self, path, start, stop, label):
        info = audioinfo(path)
        sr = info.samplerate
        sig, ssr = audioread(path, start=int(start*sr), stop=int(stop*sr))
        return SpeechActivity(sig, ssr, speechtype=label)

    def __getitem__(self, idx):
        path, ((ts, te), label) = self._allsegs[idx]
        samp = self.read(os.path.join(self.audroot, path), ts, te, label)
        if self.transform:
            samp = self.transform(samp)

        return samp

    def __len__(self):
        return len(self._allsegs)

    def __repr__(self):
        """Representation of RATS_SAD."""
        return r"""{}({}, sr={}, transform={})
        """.format(self.__class__.__name__, self.root, self.sr, self.transform)

    def __str__(self):
        """Print out a summary of instantiated dataset."""
        ss, ns = 0, 0  # speech and nonspeech durations
        for path, seg in self._allsegs:
            (ts, te), label = seg
            if label == 's':
                ss += (te-ts)
            else:
                ns += (te-ts)
        report = """
            +++++ Summary for [{}] partition [{}] +++++
            Total [{}] valid segments to be processed.
            Total speech duration: [{:.2f}] hours
            Total nonspeech duration: [{:.2f}] hours
        """.format(self.__class__.__name__, self.partition,
                   len(self._allsegs), ss/3600, ns/3600)

        return report

    @staticmethod
    def tab2segs(tabpath):
        """Read a .tab file to a list of segments.

        An example line:
            dev-1	19356_urd_src	0	3.38	NS	manual urd	original
        """
        segs = []
        with open(tabpath, 'r') as fp:
            for line in fp:
                line = line.rstrip().split()
                stype, tstart, tend = line[4], float(line[2]), float(line[3])
                segs.append(((tstart, tend), stype.lower()))
        return segs
