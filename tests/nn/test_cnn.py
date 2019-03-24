"""Test CNN modules."""
from torch import rand

from audlib.nn.cnn import STRFLayer


def test_strflayer():
    """Test STRFLayer."""
    layer = STRFLayer(100, 24, 1, 1.5, 16, (100, 100), (4, 4), 5, 32)
    assert layer(rand(1, 100, 100)).detach().numpy().shape == (1, 32, 4, 4)
    layer = STRFLayer(100, 24, 1, 1.5, 16, (500, 100), (4, 4), 5, 32)
    assert layer(rand(1, 500, 100)).detach().numpy().shape == (1, 32, 4, 4)
    return


if __name__ == "__main__":
    test_strflayer()
