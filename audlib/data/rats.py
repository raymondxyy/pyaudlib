"""Define RATS datasets in PyTorch style."""

from pdb import set_trace

from .dataset import Dataset
import glob
from audlib.io.audio import audioread
import os
import numpy as np
import pickle
import soundfile as sf


class RATS_SAD(Dataset):
    """RATS Speech Activity Detection dataset (LDC2015S02)

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

    def __init__(self, root, subset='dev-1', channel='AH', select=None,
                 transform=None):
        """Create a dataset from `channel` and takes specified transform.

        Arguments
        ---------
        root: str
            Root directory of the RATS_SAD dataset.
        subset: str
            One of dev-1, dev-2, or train.
        channel: str
            A selection of RATS channels. Available channels are [A-H].
        transform: callable [None]
            Transform to be done on each audio sequence.

        Outputs
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
        if subset not in ('dev-1', 'dev-2', 'train'):
            raise ValueError("Error: Must be one of dev-1/dev-2/train")
        noisydir = os.path.join(root, 'data',
                                'aligned-{}'.format(subset), 'audio')
        cleandir = os.path.join(root, 'data', subset, 'audio/src')
        vaddir = os.path.join(self.root, 'data', subset, 'sad/src')
        assert os.path.exists(noisydir), \
            "Noisy directory does not exist!".format(noisydir)
        assert os.path.exists(cleandir), \
            "Clean directory does not exist!".format(cleandir)
        assert os.path.exists(vaddir), \
            "VAD directory does not exist!".format(vaddir)

        self.flist = []  # holds all files to be processed
        for chan in channel:
            for noisy in glob.iglob(os.path.join(noisydir, chan, '*.flac')):
                fid = os.path.basename(noisy)[:5]
                clean = glob.glob(os.path.join(cleandir,
                                               "{}*.flac".format(fid)))[0]
                vad = glob.glob(os.path.join(vaddir,
                                             "{}*.tab".format(fid)))[0]
                self.flist.append((noisy, clean, vad))

        self.select = select
        self.transform = transform

    def __len__(self):
        """Return number of audio files to be processed."""
        return len(self.flist)

    def __getitem__(self, idx):
        """Convert (noisy, clean, vad) paths to features on indexing."""
        noisy, clean, vad = self.flist[idx]
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
        vadref = tabread(vad)
        vadref = [(ts-offset, te-offset) for ts, te in vadref]
        sample = {'chan1': {'sr': sr1, 'data': sigx},
                  'clean': {'sr': sr1, 'data': sigs},
                  'vad': vadref}

        if self.transform:
            sample = self.transform(sample)

        return sample


def tabread(tabpath):
    """Read .tab file in RATS_SAD dataset and return speech timestamps."""
    tstamps = []
    with open(tabpath, 'r') as fp:
        lines = fp.readlines()
        for l in lines:
            l = l.strip().split()
            stype, tstart, tend = l[4], float(l[2]), float(l[3])
            if stype == 'S':  # speech
                tstamps.append((tstart, tend))
    return tstamps


def test_RATS_SAD():
    root = '/home/xyy/data/RATS_SAD'  # change to your directory

    dataset = RATS_SAD(root, channel='AH')
    print("RATS_SAD set samples: [{}]".format(len(dataset)))
    for sr, noisy, clean, vad in dataset:
        pass


if __name__ == '__main__':
    test_RATS_SAD()
