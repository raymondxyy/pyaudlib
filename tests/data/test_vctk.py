"""Test suites for VCTK."""

from audlib.data.vctk import SEVCTK2chan, SEVCTKNoRev


def test_VCTK2chan():
    root = '/home/xyy/data/VCTKsmall'  # change to your directory accordingly

    def _testset():
        db = SEVCTK2chan(root, mode='test', testdir='*')
        return len(db) == 5184

    def _trainset():
        db = SEVCTK2chan(root, mode='train')
        return len(db) == 10368

    def _validset():
        db = SEVCTK2chan(root, mode='valid')
        return len(db) == 2592

    assert _trainset()
    assert _validset()
    assert _testset()


def test_VCTKNoRev():
    root = '/home/xyy/data/VCTK'  # change to your directory accordingly

    def _testset():
        db_test = SEVCTKNoRev(root, mode='test', testdir='*')
        return len(db_test) == 44526

    def _trainset():
        db_train = SEVCTKNoRev(root, mode='train')
        return len(db_train) == 29467

    def _validset():
        db_valid = SEVCTKNoRev(root, mode='valid')
        return len(db_valid) == 6959

    assert _testset()
    assert _trainset()
    assert _validset()


if __name__ == '__main__':
    test_VCTK2chan()
    test_VCTKNoRev()
