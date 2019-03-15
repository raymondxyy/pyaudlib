"""Test suite for avspoof dataset."""
from audlib.data.avspoof import AVSpoof


def test_avspoof():
    """Test AVSpoof class."""
    dataset = AVSpoof('/home/xyy/data/AVSpoof')
    print(dataset)


if __name__ == "__main__":
    test_avspoof()
