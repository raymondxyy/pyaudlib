"""Test suite for RATS dataset."""
from audlib.data.rats import SERATS_SAD, RATS_SAD


def test_SERATS_SAD():
    root = '/home/xyy/data/RATS_SAD'  # change to your directory

    dataset = SERATS_SAD(root, channels='AH')
    assert len(dataset) == 320


def test_RATS_SAD():
    root = '/home/xyy/data/RATS_SAD'
    dataset = RATS_SAD(root, 'dev-1')
    print(dataset)
    assert len(dataset) == 391402
    dataset = RATS_SAD(root, 'dev-2', filt=lambda seg: seg[1] == 's')
    print(dataset)


if __name__ == '__main__':
    test_SERATS_SAD()
    test_RATS_SAD()
