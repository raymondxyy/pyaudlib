"""Test suite for TIMIT."""
import os
import pytest
from audlib.data.timit import TIMIT_ASR, TIMIT_SID


@pytest.mark.skipif('TIMIT_ROOT' not in os.environ,
                    reason='ENV $TIMIT_ROOT unspecified.')
def test_timit_asr():
    dataset = TIMIT_ASR(os.environ['TIMIT_ROOT'], 'train')
    assert len(dataset) == 4620
    dataset = TIMIT_ASR(os.environ['TIMIT_ROOT'], 'core-test')
    assert len(dataset) == 192
    dataset = TIMIT_ASR(os.environ['TIMIT_ROOT'], 'complete-test')
    assert len(dataset) == 1344


@pytest.mark.skipif('TIMIT_ROOT' not in os.environ,
                    reason='ENV $TIMIT_ROOT unspecified.')
def test_timit_sid():
    dataset = TIMIT_SID(os.environ['TIMIT_ROOT'])
    assert len(dataset) == 4620
    dataset = TIMIT_SID(os.environ['TIMIT_ROOT'], False)
    assert len(dataset) == 1680


if __name__ == '__main__':
    test_timit_asr()
    test_timit_sid()
