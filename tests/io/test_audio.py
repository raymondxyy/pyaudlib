"""Test suite for audio I/O."""
import os

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
    assert sig.shape[1] == 2


if __name__ == '__main__':
    test_audioread()
