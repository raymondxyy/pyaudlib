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


def sphereinfo(path):
    """Read metadata of a embedded-shorten sphere file.

    A sphere header looks like the following:
    NIST_1A
       1024
    sample_count -i 143421600
    sample_n_bytes -i 2
    channel_count -i 2
    sample_byte_format -s2 01
    sample_rate -i 16000
    sample_coding -s26 pcm,embedded-shorten-v2.00
    sample_checksum -i 24616
    end_head
    """
    info = {}
    with open(path, 'rb') as fp:
        for line in fp:
            if line.strip() == b'1024':
                break

        for line in fp:
            if line.strip() == b'end_head':
                break
            items = line.strip().decode().split()
            field, flag, val = items[0], items[1], ' '.join(items[2:])
            info[field] = int(val) if flag == '-i' else val

    return info


class SphereInfo(object):
    """soundfile.info interface for embedded-shorten."""

    def __init__(self, path):
        """Read metadata of a sphere file."""
        super(SphereInfo, self).__init__()

        info = sphereinfo(path)
        self.samplerate = info['sample_rate']
        self.frames = info['sample_count']


def audioinfo(path):
    """Read metadata of an audio file.

    A wrapper of soundfile.info plus a class for embedded-shorten files.
    Parameters
    ----------
    path: str
        Full path to audio file.

    Returns
    -------
    info: class
        A soundfile.info class of available metadata.

    """
    try:
        return sf.info(path)
    except RuntimeError:
        return SphereInfo(path)


def sphereread(path, start=0, stop=None):
    """Read a embedded-shorten .sph file using sph2pipe.

    Assume `stop` does not exceed total duration.
    """
    assert start >= 0, "Must start at non-negative sample point."
    if stop is None:
        dur = "{}:".format(start)
    else:
        dur = "{}:{}".format(start, stop)
    cmd = [_sph2pipe, '-f', 'wav', '-s', dur, path]
    x, xsr = sf.read(io.BytesIO(subprocess.check_output(cmd)))

    return x, xsr


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
        x, xsr = sphereread(path, start=start, stop=stop)

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
