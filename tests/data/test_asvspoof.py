"""Test suite for avspoof dataset."""
from audlib.data.asvspoof import ASVspoof2017
from audlib.io.audio import no_shorter_than

AVSPOOF_ROOT = '/home/xyy/data/ASVspoof2017'


def test_asvspoof2017():
    """Test AVSpoof class."""
    for partition in ['train', 'valid', 'test']:
        dataset = ASVspoof2017(AVSPOOF_ROOT, partition,
                               filt=lambda p: no_shorter_than(p, 1.5))
        print(dataset)


if __name__ == "__main__":
    test_asvspoof2017()
