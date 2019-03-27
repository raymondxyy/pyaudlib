"""Test suite for TIMIT."""
from audlib.data.timit import TIMIT


def test_timit():
    dataset = TIMIT('/home/xyy/data/timit')
    print(dataset)


if __name__ == "__main__":
    test_timit()
