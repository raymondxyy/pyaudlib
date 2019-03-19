"""Test suite for avspoof dataset."""
from audlib.data.avspoof import AVSpoof, is_genuine, is_replay

AVSPOOF_ROOT = '/home/xyy/data/AVSpoof'


def test_avspoof():
    """Test AVSpoof class."""
    dataset = AVSpoof(AVSPOOF_ROOT)
    print(dataset)
    dataset = AVSpoof(AVSPOOF_ROOT, sr=16000,
                      filt=lambda p: is_genuine(p) or is_replay(p))
    print(dataset)


if __name__ == "__main__":
    test_avspoof()
