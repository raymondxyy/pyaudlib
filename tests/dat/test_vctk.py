"""Test suites for VCTK.

# TODO:
    - [ ] VCTK2chan
        - [x] training set size 10368
        - [x] validation set size 2592
        - [x] testing set size 5184
"""

from audlib.data.vctk import VCTK2chan


def test_VCTK2chan():
    indir = '/home/xyy/data/VCTKsmall'  # change to your directory accordingly

    def _testset():
        db = VCTK2chan(indir, mode='test', testdir='*')
        print("Entire test set samples: [{}]".format(len(db)))
        return len(db) == 5184

    def _trainset():
        db = VCTK2chan(indir, mode='train')
        print("Training set samples: [{}]".format(len(db)))
        return len(db) == 10368

    def _validset():
        db = VCTK2chan(indir, mode='valid')
        print("Validation set samples: [{}]".format(len(db)))
        return len(db) == 2592

    assert _trainset()
    assert _validset()
    assert _testset()


if __name__ == '__main__':
    test_VCTK2chan()
