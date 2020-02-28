"""Some functions to deal with I/O of different audio file formats."""
import sys
from random import randrange
import soundfile as sf
import numpy as np
import os
import subprocess
import io
import platform


class WorkingDirectory:
    def __init__(self, dirpath):
        self.old_dirpath = os.getcwd()
        self.dirpath = dirpath

    def __enter__(self):
        os.chdir(self.dirpath)
        return self

    def __exit__(self, type, value, traceback):
        os.chdir(self.old_dirpath)


# Global variables used in this module
_sph2pipe = os.path.join(
    os.path.dirname(__file__),
    '../tools/sph2pipe/',
    'sph2pipe.exe' if platform.system() == 'Windows' else 'sph2pipe')

# Compile if not exist
if not os.path.exists(_sph2pipe):
    try:
        subprocess.check_output(['gcc', '--version'])
    except OSError:
        print('''gcc is not installed on your system or it is not in PATH.
    Please, make sure gcc is accessible through PATH.''', file=sys.stderr)

    file_dir = os.path.dirname(os.path.realpath(__file__))
    build_dir = os.path.join(file_dir, '..', 'tools', 'sph2pipe')
    with WorkingDirectory(build_dir):
        gcc_cmd = [
            'gcc',
            # Remove sph2pipe-specific warnings
            '-Wno-implicit-function-declaration',
            '-Wno-pointer-sign',
            '-Wno-format',
            '-Wno-implicit-int',
            '-Wno-return-type',
            # Set the output to sph2pipe
            '-o',
            'sph2pipe',
            # Need to list all .c files here
            'sph2pipe.c',
            'file_headers.c',
            'shorten_x.c',
            '-lm']
        subprocess.check_call(gcc_cmd)
        print("sph2pipe compiled!")


def sphereinfo(path):
    """Read metadata of a embedded-shorten sphere file."""
    return SphereInfo(path)


class SphereInfo(object):
    """soundfile.info interface for embedded-shorten-v2.00.

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
    __slots__ = 'samplerate', 'frames'

    def __init__(self, path):
        """Read metadata of a sphere file."""
        super(SphereInfo, self).__init__()
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

        self.samplerate = info['sample_rate']
        self.frames = info['sample_count']


def audioinfo(path):
    """A wrapper of soundfile.info plus a class for embedded-shorten files.

    Returns
    -------
    info: soundfile._SoundFileInfo or SphereInfo

    """
    try:
        return sf.info(path)
    except RuntimeError:
        return SphereInfo(path)


def sphereread(path, frames=-1, start=0, stop=None, **kwargs):
    """Read a embedded-shorten .sph file using sph2pipe.

    TODO: Make padding work.
    """
    assert start >= 0, "Must start at non-negative sample point."
    if stop is not None:
        assert stop > start, "Invalid stop."
        dur = f"{start}:{stop}"
    elif frames > -1:
        dur = f"{start}:{start+frames}"
    else:
        dur = f"{start}:"

    cmd = [_sph2pipe, '-f', 'wav', '-s', dur, path]
    x, sr = sf.read(io.BytesIO(subprocess.check_output(cmd)), **kwargs)

    return x, sr


def audioread(file, frames=-1, start=0, stop=None, **kwargs):
    """A wrapper for soundfile.read plus ability to read .sph file.

    Parameters
    ----------
    path: str or int or file-like object
        File path or object to read from. See soundfile.SoundFile for details.

    See Also
    --------
    soundfile.read

    """
    try:
        x, sr = sf.read(file, frames, start, stop, **kwargs)
    except RuntimeError:  # fix for sph pcm-embedded shortened v2
        assert type(file) is str, ".sph file only accepts string input."
        x, sr = sphereread(file, frames, start, stop, **kwargs)

    return x, sr


def audiowrite(file, data, sr, norm=True, **kwargs):
    """A wrapper of soundfile.write with normalization option.

    Parameters
    ----------
    file: str or int or file-like object
        The file to write to. See soundfile.SoundFile.
    data: array_like
        Audio waveform.
    sr: int
        Output sampling rate.
    norm: bool, optional
        Normalize amplitude by scaling so that maximum absolute amplitude is 1.
        Default to true.

    """
    absmax = np.max(np.abs(data))  # in case all entries are 0s
    if norm and (absmax != 0):
        data /= absmax

    return sf.write(file, data, sr, **kwargs)


def chk_duration(path, minlen=None, maxlen=None, unit='second'):
    """Check if audio from path satisfies duration requirement.

    Parameters
    ----------
    path: str
        File path to audio.
    minlen: float, optional
        Inclusive minimum length of selection in seconds or samples.
        Default to any duration.
    maxlen: float, optional
        Exclusive maximum length of selection in seconds or samples.
        Default to any duration.
    unit: str, optional
        The unit in which `minlen` and `maxlen` are interpreted.
        Options are:
            - 'second' (default)
            - 'sample'

    Returns
    -------
    okay: bool
        True if all conditions are satisfied. False otherwise.

    """
    if (minlen is None) and (maxlen is None):
        return True

    info = audioinfo(path)
    sr, sigsize = info.samplerate, info.frames
    if unit == 'second':
        minlen = int(minlen * sr) if (minlen is not None) else None
        maxlen = int(maxlen * sr) if (maxlen is not None) else None
    if minlen is not None and (sigsize < minlen):
        return False
    if maxlen is not None and (sigsize >= maxlen):
        return False

    return True


def shorter_than(path, duration, unit='second'):
    """Check if audio is shorter than duration in unit."""
    return chk_duration(path, maxlen=duration, unit=unit)


def no_shorter_than(path, duration, unit='second'):
    """Check if audio is not shorter than duration in unit."""
    return not shorter_than(path, duration, unit=unit)


def longer_than(path, duration, unit='second'):
    """Check if audio is longer than duration in unit."""
    if unit == 'sample':
        duration += 1
    return chk_duration(path, minlen=duration, unit=unit)


def no_longer_than(path, duration, unit='second'):
    """Check if audio is not longer than duration in unit."""
    return not longer_than(path, duration, unit=unit)


def randsel(path, minlen=0, maxlen=None, unit="second"):
    """Randomly select a portion of audio from path.

    Parameters
    ----------
    path: str
        File path to audio.
    minlen: float, optional
        Inclusive minimum length of selection in seconds or samples.
    maxlen: float, optional
        Exclusive maximum length of selection in seconds or samples.
    unit: str, optional
        The unit in which `minlen` and `maxlen` are interpreted.
        Options are:
            - 'second' (default)
            - 'sample'

    Returns
    -------
    tstart, tend: tuple of int
        integer index of selection

    """
    info = audioinfo(path)
    sr, sigsize = info.samplerate, info.frames
    if unit == 'second':
        minoffset = int(minlen*sr)
        maxoffset = int(maxlen*sr) if maxlen else sigsize
    else:
        minoffset = minlen
        maxoffset = maxlen if maxlen else sigsize

    assert (minoffset < maxoffset) and (minoffset <= sigsize), \
        f"""BAD: siglen={sigsize}, minlen={minoffset}, maxlen={maxoffset}"""

    # Select begin sample
    tstart = randrange(max(1, sigsize-minoffset))
    tend = randrange(tstart+minoffset, min(tstart+maxoffset, sigsize+1))

    return tstart, tend


def randread(fpath, minlen=None, maxlen=None, unit='second'):
    """Randomly read a portion of audio from file."""
    nstart, nend = randsel(fpath, minlen, maxlen, unit)
    return audioread(fpath, start=nstart, stop=nend)


def fixread(fpath, duration, unit='second'):
    """Randomly read a fix-length portion of audio from file."""
    if unit == 'second':  # convert to samples
        duration = int(duration * audioinfo(fpath).samplerate)
    nstart, nend = randsel(fpath, minlen=duration, maxlen=duration+1,
                           unit='sample')
    return audioread(fpath, start=nstart, stop=nend)
