"""Datasets derived from the TIMIT dataset for phoneme recognition."""
import os
from random import randrange

from ..io.audio import audioinfo
from .dataset import AudioDataset, audioread
from .datatype import Audio

# Phoneme table defined in TIMIT's doc
PHONETABLE = {p: i for i, p in enumerate(
    """aa ae ah ao aw ax ax-h axr ay b bcl ch d dcl dh dx eh el em en
        eng epi er ey f g gcl h# hh hv ih ix iy jh k kcl l m n ng nx ow
        oy p pau pcl q r s sh t tcl th uh uw ux v w y z zh""".split()
    )}
VOWELS = """iy ih eh ey ae aa aw ay ah ao oy ow uh uw ux er ax ix axr
            ax-h""".split()
SEMIVOWELS = "l r w y hh hv el".split()
STOPS = "b d g p t k dx q".split()
AFFRICATES = "jh ch".split()
FRICATIVES = "s sh z zh f th v dh".split()
NASALS = "m n ng em en eng nx".split()
OTHERS = "pau epi h# 1 2".split()


def phnread(path):
    """Read a .PHN (or .WRD) file.

    Format: <BEGIN-SAMPLE> <ENDING-SAMPLE> <PHONE>
    Example:
        0 3050 h#
        3050 4559 sh
        4559 5723 ix
    """
    try:
        seq = []
        with open(path) as fp:
            for line in fp:
                ns, ne, ph = line.rstrip().split()
                seq.append(((int(ns), int(ne)), ph))
        return seq
    except FileNotFoundError:
        print(f"[{path}] does not exist!")
        return None


def txtread(path):
    """Read a .TXT transcript file.

    Format: <BEGIN-SAMPLE> <END-SAMPLE> <TRANSCRIPT>
    Example:
        0 46797 She had your dark suit in greasy wash water all year.
    """
    try:
        with open(path) as fp:
            for line in fp:
                line = line.rstrip().split()
                ns, ne, tr = line[0], line[1], ' '.join(line[2:])
                transcrpt = (int(ns), int(ne)), tr
        return transcrpt
    except FileNotFoundError:
        print(f"{path} does not exist!")
        return None


def spkrinfo(path, istrain):
    """Load speaker table from file.

    Format:
        ID  Sex DR Use  RecDate   BirthDate  Ht    Race Edu  Comments
    Example:
        ABC0  M  6  TRN  03/03/86  06/17/60  5'11"  WHT  BS

    Parameters
    ----------
    path: str
        Path to SPRKINFO.TXT.
    istrain: bool
        Retrieve speakers from either train or test set.

    Return
    ------
    type: dict[str] -> int

    """
    with open(path) as fp:
        spkrt = {}
        ii = 0  # for label
        for line in fp:
            if line[0] != ';':  # ignore header
                line = line.rstrip().split()
                sid, train = line[0], line[3].upper() == 'TRN'
                if not istrain ^ train:
                    spkrt[sid] = ii
                    ii += 1
    return spkrt


class TIMITSpeech(Audio):
    """A data structure for TIMIT audio."""
    __slots__ = 'speaker', 'gender', 'transcript', 'phonemeseq', 'wordseq'

    def __init__(self, signal=None, samplerate=None, speaker=None, gender=None,
                 transcript=None, phonemeseq=None, wordseq=None):
        super(TIMITSpeech, self).__init__(signal, samplerate)
        self.speaker = speaker  # str
        self.gender = gender  # str
        self.transcript = transcript  # str
        self.phonemeseq = phonemeseq  # [((sample-start, sample-end), str)]
        self.wordseq = wordseq  # [((sample-start, sample-end), str)]


class TIMIT(AudioDataset):
    """Generic TIMIT dataset for phoneme recognition.

    The dataset should follow the directory structure below:
    root
    ├── CONVERT
    ├── SPHERE
    └── TIMIT
        ├── DOC
        ├── TEST
        │   ├── DR1
        │   ├── DR2
        │   ├── DR3
        │   ├── DR4
        │   ├── DR5
        │   ├── DR6
        │   ├── DR7
        │   └── DR8
        └── TRAIN
            ├── DR1
            ├── DR2
            ├── DR3
            ├── DR4
            ├── DR5
            ├── DR6
            ├── DR7
            └── DR8
    """
    @staticmethod
    def isaudio(path):
        return path.upper().endswith('.WAV')

    def __init__(self, root, task=None, train=True, filt=None, transform=None):
        """Instantiate an ASVspoof dataset.

        Parameters
        ----------
        root: str
            The root directory of TIMIT.
        task: str, None
            Convert string label to integer labels if specified.
            One of [speaker/phoneme].
        train: bool, True
            Retrieve training or test set.
        filt: callable, None
            Filters to be applied on each audio path. Default to None.
        transform: callable(TIMITSpeech) -> TIMITSpeech
            User-defined transformation function.

        Returns
        -------
        An class instance `TIMIT` that has the following properties:
            - len(TIMIT) == number of usable audio samples
            - TIMIT[idx] == a TIMITSpeech instance

        See Also
        --------
        TIMITSpeech, dataset.AudioDataset, datatype.Audio

        """
        self.train = train
        self._spkr_table = spkrinfo(
            os.path.join(root, 'TIMIT/DOC/SPKRINFO.TXT'), train)
        if train:
            audroot = os.path.join(root, 'TIMIT/TRAIN')
        else:
            audroot = os.path.join(root, 'TIMIT/TEST')

        def _2label(sample):
            """Convert true labels to integer labels for specific tasks."""
            if task == 'speaker':
                return self.spkr2label(sample)
            elif task == 'phoneme':
                return self.phon2label(sample)
            elif task is None:
                return sample
            else:
                raise NotImplementedError

        def _read(path):
            """Different options to read an audio file."""
            pbase = os.path.splitext(path)[0]
            gsid = pbase.split('/')[-2]
            gender, sid = gsid[0], gsid[1:]
            assert sid in self._spkr_table
            phoneseq = phnread(pbase+'.PHN')
            wrdseq = phnread(pbase+'.WRD')
            transcrpt = txtread(pbase+'.TXT')
            return _2label(
                TIMITSpeech(*audioread(path), speaker=sid, gender=gender,
                            transcript=transcrpt, phonemeseq=phoneseq,
                            wordseq=wrdseq))

        super(TIMIT, self).__init__(
            audroot, filt=self.isaudio if not filt else lambda p:
                self.isaudio(p) and filt(p), read=_read, transform=transform)

    def spkr2label(self, sample):
        """Replace sample.speaker by its integer ID."""
        sample.speaker = self._spkr_table[sample.speaker]
        return sample

    def phon2label(self, sample):
        """Replace sample.phonemeseq by its integer ID."""
        sample.phonemeseq = [
            (t, PHONETABLE[p]) for t, p in sample.phonemeseq]
        return sample

    def __repr__(self):
        """Representation of TIMIT."""
        return r"""{}({}, transform={})
        """.format(self.__class__.__name__, self.root, self.transform)

    def __str__(self):
        """Print out a summary of instantiated dataset."""
        spkr_appeared = set([])
        for path in self._filepaths:
            sid = path.split('/')[-2][1:]
            assert sid in self._spkr_table, f"{sid} not a valid speaker!"
            spkr_appeared.add(sid)
        phoncts = {p: 0 for p in PHONETABLE}
        mindur = {p: 100 for p in PHONETABLE}
        maxdur = {p: 0 for p in PHONETABLE}
        for path in self._filepaths:
            sr = audioinfo(os.path.join(self.root, path)).samplerate
            path = os.path.join(self.root, os.path.splitext(path)[0]+'.PHN')
            for (ts, te), pp in phnread(path):
                assert pp in phoncts, f"[{pp}] not in phone dict!"
                phoncts[pp] += 1
                dur = (te - ts) / sr * 1000
                if mindur[pp] > dur:
                    mindur[pp] = dur
                if maxdur[pp] < dur:
                    maxdur[pp] = dur
        totcts = sum(v for p, v in phoncts.items())
        report = """
            +++++ Summary for [{}] partition [{}] +++++
            Total [{}] valid files to be processed.
            Total [{}/{}] speakers appear in this set.
            [Phoneme]: [counts], [percentage], [min-max duration (ms)]\n{}
        """.format(self.__class__.__name__, 'train' if self.train else 'test',
                   len(self._filepaths), len(spkr_appeared),
                   len(self._spkr_table),
                   "\n".join(
                       f"\t\t[{p:>4}]: [{c:>4}], [{c*100/totcts:2.2f}%], [{mindur[p]:.1f}-{maxdur[p]:.0f}]"
                       for p, c in phoncts.items()))

        return report


# Some filter functions
def isvowel(phone, semivowels=True):
    """Check if phone is a vowel (or a semivowel)."""
    if semivowels:
        return (phone in VOWELS) or (phone in SEMIVOWELS)

    return phone in VOWELS


def isspeech(phone):
    """Check if phone belongs to speech."""
    return phone not in OTHERS


def utt_no_shorter_than(path, duration, unit='second'):
    """Check for utterance duration after silence removal."""
    pbase = os.path.splitext(path)[0]
    phonseq = phnread(pbase+'.PHN')
    dur = sum(te-ts if isspeech(p) else 0 for (ts, te), p in phonseq)
    if unit == 'second':
        return dur >= duration * audioinfo(path).samplerate
    elif unit == 'sample':
        return dur >= duration
    else:
        raise ValueError(f"Unsupported unit: {unit}.")


# Some transform functions
def rmsilence(sample):
    """Remove silence from the waveform of a sample."""
    ns, ne = sample.wordseq[0][0][0], sample.wordseq[-1][0][1]
    return sample.signal[ns:ne]


def randselwave(sample, minlen=0, maxlen=None, nosilence=True):
    """Randomly select a portion of the signal from a sample."""
    if nosilence:
        sig = rmsilence(sample)
    else:
        sig = sample.signal

    sigsize = len(sig)
    minoffset = int(minlen * sample.samplerate)
    maxoffset = min(int(maxlen*sample.samplerate),
                    sigsize) if maxlen else sigsize

    assert (minoffset < maxoffset) and (minoffset <= sigsize), \
        f"""BAD: siglen={sigsize}, minlen={minoffset}, maxlen={maxoffset}"""

    # Select begin sample
    ns = randrange(max(1, sigsize-minoffset))
    ne = randrange(ns+minoffset, min(ns+maxoffset, sigsize+1))

    return sig[ns:ne]


def randselphon(sample, phonfunc=None):
    """Randomly select the waveform corresponding to a single phone.

    Keyword Parameters
    ------------------
    phonfunc: callable(str) -> bool, None
        A filter function on phone.

    """
    (ns, ne), ph = sample.phonemeseq[randrange(len(sample.phonemeseq))]
    if phonfunc is not None:
        while not phonfunc(ph):
            (ns, ne), ph = sample.phonemeseq[randrange(len(sample.phonemeseq))]

    return sample.signal[ns:ne], ph
