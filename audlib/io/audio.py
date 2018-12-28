"""Some functions to deal with I/O of different audio file formats."""

import soundfile as sf
from resampy import resample
import numpy as np
import os
import subprocess
import io

# Global variables used in this module
_sph2pipe = os.path.dirname(__file__)+'/../../tools/sph2pipe/sph2pipe'

assert os.path.exists(_sph2pipe)


def audioread(path, sr=None, start=0, stop=None, force_mono=False,
              norm=False, verbose=False):
    """Read audio from path and return an numpy array.

    Parameters
    ----------
    path: str
        path to audio on disk.
    sr: int, optional
        Sampling rate. Default to None.
        If None, do nothing after reading audio.
        If not None and different from sr of the file, resample to new sr.
    force_mono: bool
        Set to True to force mono output.
    verbose: bool
        Enable verbose.

    """
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise ValueError("[{}] does not exist!".format(path))
    try:
        x, xsr = sf.read(path, start=start, stop=stop)
    except RuntimeError:  # fix for sph pcm-embedded shortened v2
        if verbose:
            print('WARNING: Audio type not supported. Trying sph2pipe...')
        wavbytes = subprocess.check_output([_sph2pipe, '-f', 'wav', path])
        x, xsr = sf.read(io.BytesIO(wavbytes), start=start, stop=stop)

    if force_mono and (len(x.shape) > 1):
        x = np.sum(x, axis=1)/2.  # stereo->mono
    if norm:  # normalize maximum absolute amplitude to 1
        x /= np.max(np.abs(x))
    if (sr is not None) and (xsr != sr):  # need sample rate conversion
        x = resample(x, xsr, sr)
        return x, sr
    else:
        return x, xsr


def audiowrite(data, sr, outpath, norm=True, verbose=False):
    """Write a numpy array into an audio file.

    Parameters
    ----------
    data: array_like
        Audio waveform.
    sr: int
        Output sampling rate.
    outpath: str
        File path to the output on disk. Directory does not need to exist.
    norm: bool, optional
        Normalize amplitude by scaling so that maximum absolute amplitude is 1.
        Default to true.
    verbose: bool, optional
        Print to console. Default to false.

    """
    absmax = np.max(np.abs(data))  # in case all entries are 0s
    if norm and (absmax != 0):
        data /= absmax
    outpath = os.path.abspath(outpath)
    outdir = os.path.dirname(outpath)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if verbose:
        print("Writing to {}".format(outpath))
    sf.write(outpath, data, sr)

    return
