"""Test suite for audlib.io.batch."""
import os

from audlib.io.batch import lsfiles
from audlib.io.audio import audioinfo
import audlib

SAMPLEDIR = os.path.join(os.path.dirname(audlib.__file__), '../samples')


def test_lsfiles():
    """Test lsfiles."""
    def longer_than_3sec(fpath):
        info = audioinfo(fpath)
        return (info.frames / info.samplerate) > 3.

    assert len(lsfiles(SAMPLEDIR, relpath=True)) == 3
    assert len(lsfiles(SAMPLEDIR, filt=longer_than_3sec)) == 2


if __name__ == "__main__":
    test_lsfiles()
