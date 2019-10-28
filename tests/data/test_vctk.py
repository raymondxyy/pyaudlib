"""Test suites for VCTK."""
import os
import pytest
from audlib.data.vctk import SEVCTKNoRev


@pytest.mark.skipif('VCTK_ROOT' not in os.environ,
                    reason='ENV $VCTK_ROOT unspecified.')
def test_VCTKNoRev():
    root = os.environ['VCTK_ROOT']
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
