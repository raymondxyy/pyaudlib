"""Datasets derived from the CMU ARCTIC datbase.

According to the authors, The CMU_ARCTIC databases were constructed at the
Language Technologies Institute at Carnegie Mellon University as phonetically
balanced, US English single speaker databases designed for unit selection
speech synthesis research.

See http://festvox.org/cmu_arctic/index.html for more information.
"""
from glob import iglob

from .dataset import Dataset
from ..io.audio import audioread


class ARCTIC(Dataset):
    """Generic dataset framework CMU_ARCTIC.

    The database on disk should have the following structure:
    path/to/ARCTIC
    ├── cmu_us_bdl_arctic  <-- speaker bdl
    │   ├── COPYING
    │   ├── etc
    │   ├── orig  <-- wav directory with audio in chan1 and egg in chan2
    │   └── README
    └── cmu_us_slt_arctic  <-- speaker slt
        ├── COPYING
        ├── etc
        ├── orig
        └── README
    """

    def __init__(self, root, sr=None, egg=False, filt=None, transform=None):
        """Instantiate an ARCTIC dataset.

        Parameters
        ----------
        root: str
            The root directory of WSJ0.
        sr: int
            Sampling rate. ARCTIC is recorded at 32kHz.
        egg: bool
            Include the EGG signal at channel 2.
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
        super(ARCTIC, self).__init__()
        self.root = root
        self.egg = egg
        self.transform = transform

        self._all_files = list(filter(filt, iglob(f"{root}/*/orig/*.wav")))

        def _audioread(path):
            """Read audio as specified by user."""
            sig, ssr = audioread(path, sr=sr)
            if egg:
                out = {'sr': ssr, 'data': sig[0], 'egg': sig[1]}
            else:
                out = {'sr': ssr, 'data': sig[0]}

            return out

        self.audioread = _audioread

    @property
    def all_files(self):
        """Build valid file list."""
        return self._all_files

    def __str__(self):
        """Print out a summary of instantiated ARCTIC."""
        report = """
            +++++ Summary for [{}] +++++
            Total [{}] valid files to be processed.
        """.format(self.__class__.__name__, len(self.all_files))

        return report

    def __repr__(self):
        """Representation of ARCTIC."""
        return r"""{}({}, train={}, egg={}, transform={})
        """.format(self.__class__.__name__, self.root, self.egg, self.filt,
                   self.transform)

    def __len__(self):
        """Return number of audio files to be processed."""
        return len(self.all_files)

    def __getitem__(self, idx):
        """Get the idx-th example from the dataset."""
        sample = self.audioread(self.all_files[idx])

        if self.transform:
            sample = self.transform(sample)

        return sample
