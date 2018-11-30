# Some functions to deal with I/O of different audio file formats
# Author: Raymond Xia (yangyanx@andrew.cmu.edu)
#
# Change Log:
#   * 2018/1/1: Clean up naming to be consistent with other files.

import soundfile as sf
from resampy import resample
import numpy as np
import os
import subprocess
import io
from ..cfg import cfgload
import pdb

# Global variables used in this module
config = cfgload()['io']
__sph2pipe__ = str(config['sph2pipe'])
__tmp__ = config['sphtmp']
__support__ = ('wav', 'sph', 'flac', 'aiff')  # supported file types

assert os.path.exists(__sph2pipe__)


def audioread(path, sr=None, start=0, stop=None, force_mono=False,
              norm=False, verbose=False):
    """
    audioread: Read audio from path and return an numpy array.
    Args:
        path - path to audio on disk.
        [sr] - If None, do nothing after reading audio. If not None and
               different from sr of the file, resample using Secret Rabbit
               Code's resampling utility.
        [force_mono] - give True to force mono output.
        [verbose]    - enable verbose.
    """
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise ValueError("[{}] does not exist!".format(path))
    try:
        x, xsr = sf.read(path, start=start, stop=stop)
    except RuntimeError:  # fix for sph pcm-embedded shortened v2
        if verbose:
            print('WARNING: Audio type not supported. Trying sph2pipe...')
        wavbytes = subprocess.check_output([__sph2pipe__, '-f', 'wav', path])
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


def audiowrite(data, sr, outdir, normalize=True, verbose=False):
    """
    audiowrite: Write a numpy array into an audio file.
    Args:
        data   - audio as a numpy array.
        sr     - output sampling rate.
        outdir - file path to the output on disk. Directory does not need to
                 exist.
        [normalize] - normalize amplitude by scaling so that maximum absolute
                      amplitude is 1.
        [verbose]   - enable verbose.
    """
    absmax = np.max(np.abs(data))  # in case all entries are 0s
    if normalize and (absmax != 0):
        data /= absmax
    outpath = os.path.abspath(outdir)
    if not os.path.exists(os.path.dirname(outpath)):
        os.makedirs(os.path.dirname(outdir))
    if verbose:
        print("Writing to {}".format(outpath))
    sf.write(outpath, data, sr)
    return
