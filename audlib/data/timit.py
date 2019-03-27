"""Datasets derived from the TIMIT dataset for phoneme recognition."""
import os

from .dataset import AudioDataset, audioread
from .datatype import Audio


class TIMITSpeech(Audio):
    """A data structure for TIMIT audio."""
    __slots__ = 'speaker', 'gender', 'transcript', 'phonemeseq', 'wordseq'

    def __init__(self, signal=None, samplerate=None, speaker=None, gender=None,
                 transcript=None, phonemeseq=None, wordseq=None):
        super(TIMITSpeech, self).__init__(signal, samplerate)
        self.speaker = speaker
        self.transcript = transcript
        self.phonemeseq = phonemeseq
        self.gender = gender
        self.wordseq = wordseq


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
        return path.endswith('.WAV')

    @staticmethod
    def phnread(path):
        """Read a .PHN (or .WRD) file.

        Format:
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

    @staticmethod
    def txtread(path):
        """Read a .TXT transcript file."""
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

    def __init__(self, root, train=True, sr=None, filt=None, read=None,
                 transform=None):
        """Instantiate an ASVspoof dataset.

        Parameters
        ----------
        root: str
            The root directory of TIMIT.
        train: bool, optional
            Retrieve training partition?
        sr: int, optional
            Sampling rate in Hz. TIMIT is recorded at 16kHz.
        filt: callable, optional
            Filters to be applied on each audio path. Default to None.
        read: callable(str) -> (array_like, int), optional
            User-defined ways to read in an audio.
            Returned values are wrapped around an `SpoofedAudio` class.
        transform: callable(SpoofedAudio) -> SpoofedAudio
            User-defined transformation function.

        Returns
        -------
        An class instance `TIMIT` that has the following properties:
            - len(TIMIT) == number of usable audio samples
            - TIMIT[idx] == a SpoofedAudio instance

        See Also
        --------
        SpoofedAudio, dataset.AudioDataset, datatype.Audio

        """
        if train:
            root = os.path.join(root, 'TIMIT/TRAIN')
        else:
            root = os.path.join(root, 'TIMIT/TEST')
        self.train = train
        self._read = read
        self.sr = sr

        super(TIMIT, self).__init__(
            root, filt=self.isaudio if not filt else lambda p:
                self.isaudio(p) and filt(p),
            read=self.read, transform=transform)

    def read(self, path):
        """Parse a path into fields, and read audio.

        A path should look like:
        root/TIMIT/TRAIN/DR1/FCJF0/SA1.WAV
        """
        pbase = os.path.splitext(path)[0]
        gsid = pbase.split('/')[-2]
        gender, sid = gsid[0], gsid[1:]
        phoneseq = self.phnread(pbase+'.PHN')
        wrdseq = self.phnread(pbase+'.WRD')
        transcrpt = self.txtread(pbase+'.TXT')

        if not self._read:
            sig, ssr = audioread(path, sr=self.sr)
        else:
            sig, ssr = self._read(path)

        return TIMITSpeech(sig, ssr, speaker=sid, gender=gender,
                           transcript=transcrpt, phonemeseq=phoneseq,
                           wordseq=wrdseq)

    def __repr__(self):
        """Representation of TIMIT."""
        return r"""{}({}, sr={}, transform={})
        """.format(self.__class__.__name__, self.root, self.sr, self.transform)

    def __str__(self):
        """Print out a summary of instantiated dataset."""
        report = """
            +++++ Summary for [{}] partition [{}] +++++
            Total [{}] valid files to be processed.
        """.format(self.__class__.__name__, 'train' if self.train else 'test',
                   len(self._filepaths))

        return report
