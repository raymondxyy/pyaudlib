"""Test suite for TIMIT."""
import os
import pytest
from audlib.data.timit import TIMIT, utt_no_shorter_than, randselwave, \
    randselphon, isvowel


@pytest.mark.skipif('TIMIT_ROOT' not in os.environ,
                    reason='ENV $TIMIT_ROOT unspecified.')
def test_timit():
    #TODO
    SI = TIMIT(os.environ['TIMIT_ROOT'],
               filt=lambda p: 'SI' in os.path.basename(p).upper() and
                              utt_no_shorter_than(p, 5),
               transform=randselwave)
    sample = SI[0]
    print(sample)

    SI = TIMIT(os.environ['TIMIT_ROOT'],
               filt=lambda p: 'SI' in os.path.basename(p).upper() and
                              utt_no_shorter_than(p, 5),
               transform=lambda s: randselphon(s, isvowel))
    sample = SI[0]
    print(sample)
