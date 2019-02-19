"""Test suite for audio I/O."""
import os

import numpy as np

from audlib.io.audio import audioread

HOME = os.path.abspath(os.path.dirname(os.path.abspath(__file__))+"../../../")


def test_audioread():
    """Test audioread."""
    # Test mono
    path = os.path.join(HOME, 'samples/welcome16k.wav')
    sig, sr = audioread(path)

    # Test stereo
    path = os.path.join(HOME, 'samples/arctic_a0001.wav')
    sig, sr = audioread(path)
    assert sig.shape[0] == 2

    # Test mixing stereo to mono
    sig2, sr = audioread(path, force_mono=True)
    assert np.allclose(sig2, np.mean(sig, axis=0))

    return


if __name__ == '__main__':
    test_audioread()
