"""Test the ARCTIC dataset."""
import os
import pytest
from audlib.data.arctic import ARCTIC


@pytest.mark.skipif('ARCTIC_ROOT' not in os.environ,
                    reason='ENV $ARCTIC_ROOT unspecified.')
def test_arctic():
    arctic = ARCTIC(os.environ['ARCTIC_ROOT'], egg=True)
    print(arctic)
