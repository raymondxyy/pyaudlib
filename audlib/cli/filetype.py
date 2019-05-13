"""Compound file definition."""


class Audio(object):
    # TODO
    """A small class holding an audio signal.

    Available fields:
        filename - file name
        sr       - sampling rate
        sig      - raw waveform OR other representations
    """

    def __init__(self, filepath=None, sr=None, sig=None, sigtype=None,
                 savemode=None):
        """Specify file name, sampling rate, and signal here."""
        super(Audio, self).__init__()
        self.path = filepath
        self.sr = sr
        self.sig = sig
        self.sigtype = sigtype
        self.savemode = savemode


# Signal type definition
_sigtype = 'AUDIO', 'FEAT'
SIGTYPE = dict(zip(_sigtype, range(len(_sigtype))))

# Save mode definition
_savemode = 'SINGLE', 'BATCH'
SAVEMODE = dict(zip(_savemode, range(len(_savemode))))
