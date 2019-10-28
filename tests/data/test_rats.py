"""Test suite for RATS dataset."""
import os
import pytest
from audlib.data.rats import SERATS_SAD, RATS_SAD


@pytest.mark.skipif('RATS_SAD_ROOT' not in os.environ,
                    reason='ENV $RATS_SAD_ROOT unspecified.')
def test_SERATS_SAD():
    dataset = SERATS_SAD(os.environ['RATS_SAD_ROOT'], channels='AH')
    assert len(dataset) == 320


@pytest.mark.skipif('RATS_SAD_ROOT' not in os.environ,
                    reason='ENV $RATS_SAD_ROOT unspecified.')
def test_RATS_SAD():
    dataset = RATS_SAD(os.environ['RATS_SAD_ROOT'], 'dev-1')
    assert len(dataset) == 391402
