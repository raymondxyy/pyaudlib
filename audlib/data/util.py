"""Some utility functions for pre-processing audio dataset."""
from random import randrange

from ..io.audio import audioinfo


def randsel(fpath, minlen=None, maxlen=None):
    """Randomly select a portion of audio from path.

    Parameters
    ----------
    fpath: str
        file path to audio
    [minlen]: float
        minimum length of selection in seconds
    [maxlen]: float
        maximum length of selection in seconds

    Returns
    -------
    tstart, tend: tuple of int
        integer index of selection

    """
    info = audioinfo(fpath)
    sr, sigsize = info.samplerate, info.frames
    if minlen is None:
        minoffset = 0
    else:
        minoffset = int(minlen*sr)
        assert minoffset < sigsize, "`minlen` exceeding total length!"
    if maxlen is None:
        maxoffset = sigsize
    else:
        maxoffset = int(maxlen*sr)
    # Select begin sample
    tstart = randrange(max(1, sigsize-minoffset))
    tend = randrange(tstart+minoffset, min(tstart+maxoffset, sigsize))
    return tstart, tend
