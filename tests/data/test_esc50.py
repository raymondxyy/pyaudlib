"""Test suite for ESC50 class."""
import os
import pytest
from audlib.data.esc50 import ESC50


@pytest.mark.skipif('ESC50_ROOT' not in os.environ,
                    reason='ENV $ESC50_ROOT unspecified.')
def test_esc50():
    dataset = ESC50(os.environ['ESC50_ROOT'])
    print(dataset)
    return


if __name__ == "__main__":
    test_esc50()
