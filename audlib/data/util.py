"""Some utility functions for pre-processing audio dataset."""
from random import randrange, randint

import numpy as np

from ..io.audio import audioinfo, audioread


def chk_duration(fpath, minlen=None, maxlen=None):
    """Check if audio from path satisfies duration requirement.

    Parameters
    ----------
    fpath: str
        file path to audio
    minlen: float, optional
        Minimum length of selection in seconds.
        Default to any duration.
    [maxlen]: float, optional
        Maximum length of selection in seconds.
        Default to any duration.

    Returns
    -------
    okay: bool
        True if all conditions are satisfied. False otherwise.

    """
    if minlen is None and maxlen is None:
        return True

    info = audioinfo(fpath)
    sr, sigsize = info.samplerate, info.frames
    if minlen is not None and (sigsize / sr < minlen):
        return False
    if maxlen is not None and (sigsize / sr > maxlen):
        return False
    return True


def randsel(audobj, minlen=0, maxlen=None):
    """Randomly select a portion of audio from path.

    Parameters
    ----------
    audobj: str
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
    if type(audobj) is str:
        info = audioinfo(audobj)
        sr, sigsize = info.samplerate, info.frames
        minoffset = int(minlen*sr)
        maxoffset = int(maxlen*sr) if maxlen else sigsize
    elif type(audobj) is np.ndarray:
        sigsize = len(audobj)
        minoffset = int(minlen)
        maxoffset = int(maxlen) if maxlen else sigsize
    else:
        raise NotImplementedError

    assert (minoffset < maxoffset) and (minoffset < sigsize), \
        f"""siglen={sigsize}, minlen = {minoffset}, maxlen = {maxoffset}
            is bad specification."""

    # Select begin sample
    tstart = randrange(max(1, sigsize-minoffset))
    tend = randrange(tstart+minoffset, min(tstart+maxoffset, sigsize))
    return tstart, tend


def randread(fpath, sr=None, minlen=None, maxlen=None):
    """Randomly read a portion of audio from file."""
    nstart, nend = randsel(fpath, minlen, maxlen)
    return audioread(fpath, sr=sr, start=nstart, stop=nend)


def mix(sigs, sign, sr, snr):
    """Additively mix clean signal and noise at a specific SNR."""
    # TODO: Need to either update or remove this.
    noisy = np.zeros_like(sigs)
    clean = np.zeros_like(sigs)
    vad = []
    if snr == np.inf:  # pure speech
        noisy[:] = sigs
        clean[:] = noisy
        vad.append((0., len(sigs)*1./sr))
    elif snr == -np.inf:  # pure noise
        noisy[:] = sign
    else:  # numerical SNRs
        # Randomly choose if noise or speech spans entire range.
        # This will create transition of SNR:
        #   +inf --> finite SNR speech goes first
        #   -inf --> finite SNR noise goes first
        # make sure at least half of entire duration is interfered
        nstart = randint(0, len(sigs)//2)
        nend = randint(nstart+len(sigs)//2, len(sigs))
        if randint(0, 1):  # speech goes first
            noisy[:] = sigs
            # Compute segmental SNR and add noise
            se = np.sum(sigs[nstart:nend]**2)  # signal energy
            ne = np.sum(sign[nstart:nend]**2)  # noise energy
            scale = np.sqrt(se/ne / (10**(snr/10.)))
            noisy[nstart:nend] += sign[nstart:nend]*scale
            clean[:] = sigs
            vad.append((0, len(sigs)*1./sr))
        else:  # noise goes first
            noisy[:] = sign
            # Compute segmental SNR and add speech
            se = np.sum(np.abs(sigs[nstart:nend])**2)  # signal energy
            ne = np.sum(np.abs(sign[nstart:nend])**2)  # noise energy
            scale = np.sqrt(ne/se * (10**(snr/10.)))
            noisy[nstart:nend] += sigs[nstart:nend]*scale
            clean[nstart:nend] = sigs[nstart:nend]*scale
            vad.append((nstart*1./sr, nend*1./sr))

    return noisy, clean, vad
