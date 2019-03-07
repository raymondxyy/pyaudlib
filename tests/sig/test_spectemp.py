"""Test spectro-temporal functions."""
import numpy as np
from audlib.sig.spectemp import strf


def test_strf():
    frate = 100
    bins_per_octave = 12
    time_support = .2
    freq_support = 1
    phi, theta = 0.5, 0
    kdn, kup = strf(time_support, freq_support, frate, bins_per_octave,
                    phi=phi*np.pi, theta=theta*np.pi)

    return


if __name__ == "__main__":
    test_strf()
