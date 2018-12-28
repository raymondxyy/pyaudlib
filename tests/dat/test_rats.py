"""Test suite for RATS dataset."""
from audlib.data.rats import SERATS_SAD


def test_RATS_SAD():
    root = '/home/xyy/data/RATS_SAD'  # change to your directory

    dataset = SERATS_SAD(root, channels='AH')
    assert len(dataset) == 320


if __name__ == '__main__':
    test_RATS_SAD()
