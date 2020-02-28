"""Test suite for LibriSpeech."""
import os
import pytest
from audlib.data.librispeech import LibriSpeakers


@pytest.mark.skipif('LIBRISPEECH_ROOT' not in os.environ,
                    reason='ENV $LIBRISPEECH_ROOT unspecified.')
def test_librispeakers():
    #TODO
    dataset = LibriSpeakers(os.environ['LIBRISPEECH_ROOT'], (.5, .4, .1),
                            shuffle=True)
    print(dataset)



if __name__ == "__main__":
    test_librispeakers()
