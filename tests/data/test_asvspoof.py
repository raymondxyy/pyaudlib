"""Test suite for avspoof dataset."""
import os
import pytest
from audlib.data.asvspoof import ASVspoof2017
from audlib.io.audio import no_shorter_than


@pytest.mark.skipif('ASVSPOOF_ROOT' not in os.environ,
                    reason='ENV $ASVSPOOF_ROOT unspecified.')
def test_asvspoof2017():
    """Test ASVSpoof class."""
    for partition in ['train', 'valid', 'test']:
        dataset = ASVspoof2017(os.environ['ASVSPOOF_ROOT'], partition,
                               filt=lambda p: no_shorter_than(p, 1.5))
        print(dataset)


if __name__ == "__main__":
    test_asvspoof2017()
