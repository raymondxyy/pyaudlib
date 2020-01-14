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

    def is_audio(fpath): return fpath.endswith(('.wav', '.sph'))

    assert len(lsfiles(SAMPLEDIR, filt=is_audio, relpath=True)) == 2
    assert len(lsfiles(SAMPLEDIR,
                       filt=lambda p: is_audio(p) and longer_than_3sec(p))) == 1


if __name__ == "__main__":
    test_lsfiles()
