"""Test suite for TIMIT."""
import os
from audlib.data.timit import TIMIT, utt_no_shorter_than, randselwave, \
    randselphon, isvowel


def test_timit():
    SI = TIMIT('/home/xyy/data/timit',
               filt=lambda p: 'SI' in os.path.basename(p).upper() and
                              utt_no_shorter_than(p, 5),
               transform=randselwave)
    sample = SI[0]
    print(sample)

    SI = TIMIT('/home/xyy/data/timit',
               filt=lambda p: 'SI' in os.path.basename(p).upper() and
                              utt_no_shorter_than(p, 5),
               transform=lambda s: randselphon(s, isvowel))
    sample = SI[0]
    print(sample)


if __name__ == "__main__":
    test_timit()
