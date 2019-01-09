"""Test suite for nn.py."""
import torch

from audlib.nn.nn import DetLinear, DetMLP, MLP
from audlib.nn.util import printnn


def test_detlinear():
    """Test DetLinear."""
    x = torch.randn(2, 3)
    linear = DetLinear(3)
    y = linear(x)
    assert torch.allclose(x, y)


def test_detmlp():
    """Test DetMLP."""
    nhidden = 10
    featdim = 6
    x = torch.randn(2, featdim).abs_()
    mlp = DetMLP(featdim, nhidden)
    print(mlp)
    printnn(mlp)
    y = mlp(x)
    assert torch.allclose(x, y)


def test_mlp():
    """Test MLP."""
    indim = 10
    outdim = 20
    hiddims = [30, 20, 10]
    mlp = MLP(indim, outdim, hiddims)
    print(mlp)


if __name__ == '__main__':
    test_detlinear()
    test_detmlp()
    test_mlp()
