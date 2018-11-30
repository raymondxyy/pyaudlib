"""The Wall Street Journal Speech Recognition Datasets.

Contain both the pilot wsj0 and full wsj1.
"""
import glob
import os

from .dataset import SpeechRecDataset
from ..io.audio import audioread


class WSJ(SpeechRecDataset):
    """Holds common functions to be used by WSJ0 and WSJ1.

    WSJ0 and WSJ1 are required to implement `self.rootdir` attribute.
    """

    def idx2flist(self, idxpath):
        """Convert a WSJ-style .idx file to a list of data paths.

        Arguments
        ---------
        idxpath: str
            Full path to an index file (.ndx).

        Outputs
        -------
        out: list of str
            A list of strings pointing to valid audio files.
        """
        out = []
        with open(idxpath) as fp:
            for line in fp.readlines():
                if line.startswith(';;'):  # skip comment lines
                    continue
                # these are all the inconsistencies between files
                # that index names like
                # 11_3_1:wsj0/sd_tr_s/001/001c0l01.wv1
                # and actual file path like
                # 11-3.1/wsj0/sd_tr_s/001/001c0l01.wv1
                fpath = line.strip()
                fpath = fpath[:2]+'-'+fpath[3]+'.'+fpath[5]+'/'+fpath[7:]
                if os.path.exists(os.path.join(self.rootdir, fpath)):
                    out.append(fpath)
                #else:
                #    print("[{}] does not exist. Skipped.".format(fpath))
        return out

    def dot2tdict(self, dotpath):
        """Convert a .dot file to a dictionary of transcriptions.

        Arguments
        ---------
        dotpath: str
            Full path to a .dot transcription file.

        Outputs
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


class WSJ0(WSJ):
    """docstring for WSJ0."""

    def __init__(self, rootdir, mode, pattern, transmap, filt=None,
                 transform=None, verbose=False):
        """Instantiate a dataset from the WSJ0 dataset.

        Parameters
        ----------

        Returns
        -------
        """
        self.rootdir = rootdir
        self.tmap = transmap

        # Validate directories of file indices and transcriptions
        tidir = os.path.join(rootdir, "11-13.1/wsj0/doc/indices/train")
        eidir = os.path.join(rootdir, "11-13.1/wsj0/doc/indices/test")
        tddir = os.path.join(rootdir, "11-4.1/wsj0/transcrp/dots")
        eddir = os.path.join(rootdir, "11-14.1/wsj0")
        for fpath in tidir, eidir, tddir, eddir:
            self.validate_path(fpath)

        # Store all transcriptions in a dictionary
        # From a file path, retrieve its transcript with
        # self.transdict[cond][sid][uid]
        self.transdict = {}  # holds all valid transcriptions
        if mode == 'train':
            for dotpath in glob.iglob(os.path.join(tddir, '*/*/*.dot')):
                cond, sid = dotpath.split('/')[-3:-1]
                if cond not in self.transdict:
                    self.transdict[cond] = {}
                if sid not in self.transdict[cond]:
                    self.transdict[cond][sid] = {}
                self.transdict[cond][sid].update(self.dot2tdict(dotpath))
        elif mode == 'test':
            raise NotImplementedError
        elif mode == 'valid':
            raise NotImplementedError
        else:
            raise ValueError("Mode should be one of train/valid/test/!")

        # Process data files
        self.filelist = []  # holds all valid data paths
        if mode == 'train':
            for idxpath in glob.iglob(os.path.join(tidir, pattern)):
                self.filelist.extend(self.idx2flist(idxpath))
        elif mode == 'test':
            for fname in glob.iglob(os.path.join(eidir, pattern)):
                self.filelist.extend(self.idx2flist(idxpath))
        elif mode == 'valid':
            raise NotImplementedError
        else:
            raise ValueError("Mode should be one of train/valid/test/!")

        # Prepare a valid list for final output
        self.validlist = []  # holds valid file path indices
        self.oovs = {}  # holds out-of-vocab words
        for ii, fpath in enumerate(self.filelist):
            fpath = os.path.join(self.rootdir, fpath)
            if os.path.exists(fpath):
                if (filt is None) or filt(fpath):
                    cond, sid, uid = fpath.split('/')[-3:]
                    uid = uid.split('.')[0]
                    trans = self.transdict[cond][sid][uid]
                    if self.tmap.transcribable(trans):
                        self.validlist.append(ii)
                    else:
                        oov = self.tmap.trans2oov(trans)
                        for w in oov:
                            if w in self.oovs:
                                self.oovs[w] += oov[w]
                            else:
                                self.oovs[w] = oov[w]

        self.transform = transform

        if verbose:  # print dataset summary
            print("**** SUMMARY for WSJ0 *****")
            print("Total [{}] files available.".format(len(self.filelist)))
            print("Total [{}] valid files to be processed.".format(
                len(self.validlist)))
            print("Total [{}] out-of-vocabulary words".format(len(self.oovs)))
            print("\t Some examples: {}".format(
                ", ".join([e for e in self.oovs][:min(5, len(self.oovs))])))

    def __len__(self):
        """Ruturn number of audio files to be processed."""
        return len(self.filelist)

    def __getitem__(self, idx):
        """Retrieve the i-th example from the dataset."""
        fpath = os.path.join(self.rootdir, self.filelist[self.validlist[idx]])

        # Find corresponding transcript
        cond, sid, uid = fpath.split('/')[-3:]
        uid = uid.split('.')[0]
        trans = self.transdict[cond][sid][uid]

        # Convert transcript to label sequence
        label = self.tmap.trans2label(trans)

        data, sr = audioread(fpath)
        sample = {
            'sr': sr,
            'data': data,
            'trans': trans,
            'label': label
            }

        if self.transform:
            sample = self.transform(sample)

        return sample
