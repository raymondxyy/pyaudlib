"""Test suites for data.util."""

from audlib.data.util import randsel
from audlib.data.vctk import VCTK2chan


def test_randsel():
    db = VCTK2chan('/home/xyy/data/VCTKsmall', mode='test', testdir='*')
    for chan1, _ in db.flist:
        tstart, tend = randsel(chan1)
        assert tstart


if __name__ == '__main__':
    test_randsel()
