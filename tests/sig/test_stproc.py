"""Test suite for short-time processing."""
import numpy as np

from audlib.sig.window import hamming
from audlib.sig.stproc import stana, ola


def test_stproc():
    """Test the stana-ola framework."""
    sr = 16000
    sig = np.ones(10000)
    hop = .5
    wind = hamming(512, hop=hop, synth=True)

    frames = stana(sig, wind, hop, synth=True)
    sigsynth = ola(frames, wind, hop)

    assert np.allclose(sig, sigsynth[:len(sig)])


if __name__ == '__main__':
    test_stproc()
