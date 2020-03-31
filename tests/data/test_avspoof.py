"""Test suite for avspoof dataset."""
import os
import pytest
from audlib.data.avspoof import AVSpoof, is_genuine, is_replay


@pytest.mark.skipif('AVSPOOF_ROOT' not in os.environ,
                    reason='ENV $AVSPOOF_ROOT unspecified.')
def test_avspoof():
    """Test AVSpoof class."""
    dataset = AVSpoof(os.environ['AVSPOOF_ROOT'],
                      filt=lambda p: is_genuine(p) or is_replay(p))
    print(dataset)


if __name__ == "__main__":
    test_avspoof()
