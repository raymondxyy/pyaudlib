"""Test suite for TIMIT."""
import os
from audlib.data.timit import TIMIT


def test_timit():
    SI = TIMIT('/home/xyy/data/timit',
               filt=lambda p: 'SI' in os.path.basename(p).upper())
    SX = TIMIT('/home/xyy/data/timit',
               filt=lambda p: 'SX' in os.path.basename(p).upper(),
               phone=True)

    print(SX, SI)


if __name__ == "__main__":
    test_timit()
