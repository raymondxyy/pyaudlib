"""Define VCTK datasets in PyTorch style."""


import glob
from pdb import set_trace
import os

import numpy as np
import pickle
from scipy.io import wavfile
from .dataset import Dataset


class VCTKNoRev(Dataset):
    """Two-channel VCTK speech without reverberation."""

    def __init__(self, indir, subset="CT", mode="train", testdir=None,
                 transform=None, vad_aggressiveness=3):
        """Create a subset of VCTK dataset with specified transform.

        Arguments
        ---------
        indir: str
            Root VCTK directory
        subset: str ["CT"]
            Subset to use. Can be one of CT/FT.
        mode: str ["train"]
            Dataset mode. Can be one of train/valid/test.
        testdir: str [None]
            String pattern for selecting sub-directory for the test case.
        transform: callable [None]
            Transofrm to be performed on each audio sequence. If none, return
            the raw audio sequence in float on each file.
        """
        self.indir = indir
        self.subset = subset
        if subset == 'CT':
            self.name_bd = 'VCTK_CT'
        elif subset == 'FT':
            self.name_bd = 'VCTK_FT'
        else:
            raise ValueError("Error: Unknown Subset")
        self.mode = mode
        self.istest = (mode == 'test')
        self.noisydir = os.path.join(indir, "{}_{}".format(mode, subset))
        assert os.path.exists(self.noisydir)
        if self.istest:  # test needs sub-directory information
            self.testdir = testdir
        else:  # train/valid needs additional noise files
            self.noisedir = os.path.join(indir, "Noises")
            assert os.path.exists(self.noisedir)

        self.flist = []  # holds all files to be processed
        if self.istest:
            for noisy1 in glob.iglob(
                os.path.join(self.noisydir,
                             '{}/*1.wav'.format(self.testdir))):
                # Second channel path and clean reference
                noisy2 = noisy1[:-5] + '2.wav'
                # Group filenames and append to the list
                self.flist.append((noisy1, noisy2))
        else:
            for root, dirs, files in os.walk(self.noisydir):
                # Sweep first channel
                for noisy1 in glob.iglob(os.path.join(root, '*1.wav')):
                    # Second channel path and clean reference
                    bname = os.path.basename(noisy1)
                    noisy2 = noisy1[:-5] + '2.wav'
                    meta = noisy1[:-8] + '.pkl'
                    noise = os.path.join(self.noisedir,
                                         "{}_{}_1_CH1.raw".format(
                                             bname[12:15], bname[9:11]))
                    # Group filenames and append to the list
                    self.flist.append((noisy1, noisy2, meta, noise))
        self.transform = transform

    def __len__(self):
        """Return number of audio samples."""
        return len(self.flist)

    def __getitem__(self, idx):
        """Retrieve i-th path to audio sample."""
        if self.istest:  # test mode produces 2-chan audios
            noisy1, noisy2 = self.flist[idx]
            sr1, x1 = wavfile.read(noisy1)
            sr2, x2 = wavfile.read(noisy2)
            assert sr1 == sr2
            sample = {'chan1': {'sr': sr1, 'data': x1},
                      'chan2': {'sr': sr1, 'data': x2}}
        else:  # train/valid mode produces 2-chan audios + clean speech
            noisy1, noisy2, meta, noise = self.flist[idx]
            sr, x1 = wavfile.read(noisy1)
            sr2, x2 = wavfile.read(noisy2)
            assert sr == sr2

            x1 = (x1 - np.mean(x1)) * 1.0
            x2 = (x2 - np.mean(x2)) * 1.0

            x_norm = np.sqrt(np.mean(x1**2))
            x1 = x1 / x_norm
            x2 = x2 / x_norm

            with open(meta, 'rb') as fp:
                metadata = pickle.load(fp, encoding='bytes')

            n_ind = metadata[b'n_ind']
            G = metadata[b'G']
            v_norm = metadata[b'v_norm']

            n1_f = np.array(np.memmap(noise, dtype=np.dtype('<i2'), mode='r'))

            n1 = n1_f[n_ind: n_ind + len(x1)] * G / (x_norm * v_norm)
            n1 = n1 - np.mean(n1)

            # Recover clean (reverberated) speech
            xC = x1 - n1

            max_A = max(map(lambda x: max(np.abs(x)), (x1, xC)))
            x1 /= max_A
            xC /= max_A

            sample = {'chan1': {'sr': sr, 'data': x1},
                      'chan2': {'sr': sr, 'data': x2},
                      'clean': {'sr': sr, 'data': xC}
                      }

        if self.transform:
            sample = self.transform(sample)

        return sample


class VCTK2chan(Dataset):
    """New reduced-size two-channel VCTK speech dataset."""

    def __init__(self, indir, mode="train", testdir=None, select=None,
                 transform=None):
        """Create a subset of VCTK dataset with specified transform.

        Arguments
        ---------
        indir: str
            Root VCTK directory
        mode: str ["train"]
            Dataset mode. Can be one of train/valid/test.
        testdir: str [None]
            String pattern for selecting sub-directory for the test case.
        select: callable [None]
            Method to select portion of audio to be processed. Expect to
            return index tuple given a path to audio.
        transform: callable [None]
            Transofrm to be performed on each audio sequence. If none, return
            the raw audio sequence in float on each file.
        """
        self.indir = indir
        self.mode = mode
        if mode == 'train':
            self.subset = 'T'
        elif mode == 'valid':
            self.subset = 'V'
        elif mode == 'test':
            self.subset = 'E'
        else:
            raise ValueError('Subset needs to be one of train/valid/test.')
        self.istest = (mode == 'test')
        self.noisydir = os.path.join(indir, "{}_CT".format(mode))
        assert os.path.exists(self.noisydir)
        if self.istest:  # test needs sub-directory information
            self.testdir = testdir
        else:  # train/valid needs additional noise files
            self.noisedir = os.path.join(indir, "Noises")
            assert os.path.exists(self.noisedir)

        self.flist = []  # holds all files to be processed
        if self.istest:
            for noisy1 in glob.iglob(os.path.join(self.noisydir,
                                                  '{}/*1.wav'.format(self.testdir))):
                # Second channel path and clean reference
                noisy2 = noisy1[:-5] + '2.wav'
                # Group filenames and append to the list
                self.flist.append((noisy1, noisy2))
                for pp in noisy1, noisy2:
                    assert os.path.exists(pp), '{} does not exist!'.format(pp)
        else:
            for root, dirs, files in os.walk(self.noisydir):
                # Sweep first channel
                for noisy1 in glob.iglob(os.path.join(root, '*1.wav')):
                    # Second channel path and clean reference
                    bname = os.path.basename(noisy1)
                    noisy2 = noisy1[:-5] + '2.wav'
                    meta = noisy1[:-8] + '.pkl'
                    noise = os.path.join(self.noisedir,
                                         "{}_{}_{}_CH1.raw".format(
                                             bname[12:15], bname[9:11],
                                             self.subset))
                    # Group filenames and append to the list
                    self.flist.append((noisy1, noisy2, meta, noise))
                    for pp in noisy1, noisy2, meta, noise:
                        assert os.path.exists(pp), \
                            '{} does not exist!'.format(pp)
        self.transform = transform
        self.select = select

    def __len__(self):
        return len(self.flist)

    def __getitem__(self, idx):
        # Select begin and end sample
        if self.select is not None:
            tstart, tend = self.select(self.flist[idx][0])
        else:
            tstart = 0
            tend = None

        if self.istest:  # test mode produces 2-chan audios
            noisy1, noisy2 = self.flist[idx]
            sr1, x1 = wavfile.read(noisy1)
            sr2, x2 = wavfile.read(noisy2)
            assert sr1 == sr2
            sample = {'chan1': {'sr': sr1, 'data': x1[tstart:tend]},
                      'chan2': {'sr': sr1, 'data': x2[tstart:tend]}}
        else:  # train/valid mode produces 2-chan audios + clean speech
            noisy1, noisy2, meta, noise = self.flist[idx]
            sr, x1 = wavfile.read(noisy1)
            sr2, x2 = wavfile.read(noisy2)
            assert sr == sr2

            x1 = (x1 - np.mean(x1)) * 1.0
            x2 = (x2 - np.mean(x2)) * 1.0

            x_norm = np.sqrt(np.mean(x1**2))
            x1 = x1 / x_norm
            x2 = x2 / x_norm

            with open(meta, 'rb') as fp:
                metadata = pickle.load(fp, encoding='bytes')

            n_ind = metadata[b'n_ind']
            G = metadata[b'G']
            v_norm = metadata[b'v_norm']

            n1_f = np.array(np.memmap(noise, dtype=np.dtype('<i2'), mode='r'))

            n1 = n1_f[n_ind: n_ind + len(x1)] * G / (x_norm * v_norm)
            n1 = n1 - np.mean(n1)

            # Recover clean (reverberated) speech
            xC = x1 - n1
            sample = {'chan1': {'sr': sr, 'data': x1[tstart:tend]},
                      'chan2': {'sr': sr, 'data': x2[tstart:tend]},
                      'clean': {'sr': sr, 'data': xC[tstart:tend]}}

        if self.transform:
            sample = self.transform(sample)
        return sample


def test_VCTKNoRev():
    indir = '/home/xyy/data/VCTK'  # change to your directory accordingly

    def _testset():
        db_test = VCTKNoRev(indir, mode='test', testdir='*')
        print("Entire test set samples: [{}]".format(len(db_test)))
        return len(db_test) == 44526

    def _trainset():
        db_train = VCTKNoRev(indir, mode='train')
        tmp = db_train[1]
        print("Training set samples: [{}]".format(len(db_train)))
        return True

    def _validset():
        db_valid = VCTKNoRev(indir, mode='valid')
        print("Validation set samples: [{}]".format(len(db_valid)))
        return True

    assert _testset()
    assert _trainset()
    assert _validset()


def test_VCTK2chan():
    indir = '/home/xyy/data/VCTKsmall'  # change to your directory accordingly

    def _testset():
        db_test = VCTK2chan(indir, mode='test', testdir='*')
        print("Entire test set samples: [{}]".format(len(db_test)))
        return True  # len(db_test) == 44526

    def _trainset():
        db_train = VCTK2chan(indir, mode='train')
        tmp = db_train[1]
        set_trace()
        print("Training set samples: [{}]".format(len(db_train)))
        return True

    def _validset():
        db_valid = VCTK2chan(indir, mode='valid')
        print("Validation set samples: [{}]".format(len(db_valid)))
        return True

    assert _testset()
    assert _trainset()
    assert _validset()


if __name__ == '__main__':
    test_VCTKNoRev()
    test_VCTK2chan()
