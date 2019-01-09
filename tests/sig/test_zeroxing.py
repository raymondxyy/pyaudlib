"""Test suite for zero crossing utilities."""

import numpy as np

from audlib.sig.temporal import zeroxing


def test_zeroxing():
    sig = np.arange(-5, 5)
    assert np.allclose(zeroxing(sig), np.array([5]))
    assert np.allclose(zeroxing(sig, option='up'),  np.array([5]))
    assert np.allclose(zeroxing(sig, option='down'),  np.array([]))

    sig = np.arange(5, -5, -1)
    assert np.allclose(zeroxing(sig), np.array([5]))
    assert np.allclose(zeroxing(sig, option='up'),  np.array([]))
    assert np.allclose(zeroxing(sig, option='down'),  np.array([5]))

    sig = np.arange(-5, 5, 2)
    assert np.allclose(zeroxing(sig), np.array([2]))
    assert np.allclose(zeroxing(sig, interp=True),  np.array([2.5]))

    sig = np.array([-1, 0, 1, -1, 1, 0, -1])
    assert np.allclose(zeroxing(sig), np.array([1, 2, 3, 5]))
    assert np.allclose(zeroxing(sig, interp=True),  np.array([1, 2.5, 3.5, 5]))
    assert np.allclose(zeroxing(sig, option='up', interp=True),
                       np.array([1, 3.5]))

    sig = np.array([-1, 0, 0, 1])
    assert np.allclose(zeroxing(sig), np.array([]))


if __name__ == '__main__':
    test_zeroxing()
