"""Test suite for RNNs."""
import torch

from audlib.nn.rnn import ARMA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_arma():
    """Test nn.rnn.ARMA."""
    print(f"Testing ARMA with [{device}].")
    torch.enable_grad()
    G = ARMA(1, 30, 30, [10]*3).to(device)
    ins = torch.rand(10, 50, 1, requires_grad=True).to(device)
    out = G.filter(ins)
    assert ins.shape == out.shape


if __name__ == "__main__":
    test_arma()
