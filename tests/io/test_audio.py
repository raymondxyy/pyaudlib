"""Test suite for audio I/O."""
import os

import numpy as np

from audlib.io.audio import audioread, audiogen

HOME = os.path.abspath(os.path.dirname(os.path.abspath(__file__))+"../../../")


def test_audioread():
    """Test audioread."""
    # Test mono
    path = os.path.join(HOME, 'samples/welcome16k.wav')
    sig, sr = audioread(path)

    # Test stereo
    path = os.path.join(HOME, 'samples/arctic_a0001.wav')
    sig, sr = audioread(path)
    assert sig.shape[1] == 2

    # Test mixing stereo to mono
    sig2, sr = audioread(path, force_mono=True)
    assert np.allclose(sig2, np.mean(sig, axis=1))


def test_audiogen():
    """Test audiogen."""
    import numpy as np
    # Test mono
    path = os.path.join(HOME, 'samples/welcome16k.wav')
    sig, sr = audioread(path)
    siggen = audiogen(path, 512, 512)
    assert np.allclose(sig, np.concatenate(list(siggen))[:len(sig)])

    # Test stereo
    path = os.path.join(HOME, 'samples/arctic_a0001.wav')
    sig, sr = audioread(path)
    siggen = audiogen(path, 512, 512)
    assert np.allclose(sig, np.concatenate(list(siggen))[:len(sig)])

    # Test mixing stereo to mono
    siggen2 = audiogen(path, 512, 512, force_mono=True)
    assert np.allclose(np.mean(sig, axis=1),
                       np.concatenate(list(siggen2))[:len(sig)])


if __name__ == '__main__':
    test_audioread()
    test_audiogen()
