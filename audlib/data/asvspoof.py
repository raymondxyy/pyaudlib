# coding: utf-8

"""Dataset class for the ASVspoof dataset."""
import os
from .dataset import AudioDataset, audioread
from .avspoof import SpoofedAudio


class ASVspoof2017(AudioDataset):
    """Class for the ASVspoof2017 dataset.

    The dataset can be downloaded at
    https://datashare.is.ed.ac.uk/handle/10283/3055

    The dataset directory should follow the structure below.

    ASVspoof
    ├── ASVspoof2017_V2_dev
    ├── ASVspoof2017_V2_eval
    ├── ASVspoof2017_V2_train
    └── protocol_V2

    Kinnunen, Tomi; Sahidullah, Md; Delgado, Héctor; Todisco, Massimiliano;
    Evans, Nicholas; Yamagishi, Junichi; Lee, Kong Aik. (2018).
    The 2nd Automatic Speaker Verification Spoofing and Countermeasures
    Challenge (ASVspoof 2017) Database, Version 2, [sound].
    University of Edinburgh. The Centre for Speech Technology Research (CSTR).
    https://doi.org/10.7488/ds/2332.
    """

    @staticmethod
    def isaudio(path):
        return path.endswith('.wav')

    @staticmethod
    def eval_genuine(path):
        """Read and retrieve all genuine files from evalution key.

        Parameters
        ----------
        path: str
            Full path to ASVspoof2017_V2_eval.trl.txt.

        Returns
        -------
        out: list of str
            List of genuine IDs (excluding extensions).
        """
        out = []
        with open(path, 'r') as fp:
            for line in fp:
                fields = line.rstrip().split()
                ii, tt = fields[:2]
                if tt == 'genuine':
                    out.append(ii[2:-4])  # remove 'D_' and '.wav'

        return out

    @staticmethod
    def is_genuine(path, evalkey=None):
        """Return True if `path` is pointing to a genuine audio.

        `evalkey` must not be empty if path points to an evaluation file.
        """
        iid = os.path.basename(path)[:-4]
        tt, ii = iid.split('_')
        if tt == 'T':  # training
            return (int(ii) <= 1001508)
        elif tt == 'D':  # development
            return (int(ii) <= 1000760)
        elif tt == 'E':  # evaluation
            return ii in evalkey
        else:
            raise ValueError("Invalid file name.")

    def __init__(self, root, partition, filt=None, read=None, transform=None):
        """Instantiate an ASVspoof dataset.

        Parameters
        ----------
        root: str
            The root directory of AVSpoof.
        partition: str
            One of 'train', 'valid', or 'test'.
        filt: callable, optional
            Filters to be applied on each audio path. Default to None.
        read: callable(str) -> (array_like, int), optional
            User-defined ways to read in an audio.
            Returned values are wrapped around an `SpoofedAudio` class.
        transform: callable(SpoofedAudio) -> SpoofedAudio
            User-defined transformation function.

        Returns
        -------
        An class instance `avspoof` that has the following properties:
            - len(avspoof) == number of usable audio samples
            - avspoof[idx] == a SpoofedAudio instance

        See Also
        --------
        SpoofedAudio, dataset.AudioDataset, datatype.Audio

        """
        self._evalkey = None
        if partition == 'train':
            root = os.path.join(root, 'ASVspoof2017_V2_train')
        elif partition == 'valid':
            root = os.path.join(root, 'ASVspoof2017_V2_dev')
        elif partition == 'test':
            self._evalkey = self.eval_genuine(
                os.path.join(root, 'protocol_V2/ASVspoof2017_V2_eval.trl.txt'))
            root = os.path.join(root, 'ASVspoof2017_V2_eval')
        else:
            raise ValueError("partition must be one of train/valid/test.")
        self.partition = partition

        def _read(path):
            atype = None if self.is_genuine(path, self._evalkey) else 'r'
            if not read:
                sig, ssr = audioread(path)
                return SpoofedAudio(sig, ssr, attacktype=atype)
            else:
                sig, ssr = read(path)
                return SpoofedAudio(sig, ssr, attacktype=atype)

        super(ASVspoof2017, self).__init__(
            root, filt=self.isaudio if not filt else lambda p:
                self.isaudio(p) and filt(p),
            read=_read, transform=transform)

    def __repr__(self):
        """Representation of ASVspoof2017."""
        return r"""{}({}, transform={})
        """.format(self.__class__.__name__, self.root, self.transform)

    def __str__(self):
        """Print out a summary of instantiated dataset."""
        genuine, replay = 0, 0,
        for pp in self._filepaths:
            if self.is_genuine(pp, self._evalkey):
                genuine += 1
            else:
                replay += 1
        report = """
            +++++ Summary for [{}] partition [{}] +++++
            Total [{}] valid files to be processed.
            Genuine files: [{}]
            Attack files (all replays): [{}]
        """.format(self.__class__.__name__, self.partition,
                   len(self._filepaths), genuine, replay)

        return report
