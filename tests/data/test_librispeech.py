"""Test suite for LibriSpeech."""
from audlib.data.librispeech import LibriSpeakers


def test_librispeakers():
    dataset = LibriSpeakers('/home/xyy/Downloads/LibriSpeech', (.5, .4, .1),
                            shuffle=True)
    print(dataset)
    sample = dataset.trainset[0]
    return


if __name__ == "__main__":
    test_librispeakers()
