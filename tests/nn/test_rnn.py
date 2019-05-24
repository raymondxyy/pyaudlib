"""Test suite for RNNs."""
import torch

from audlib.nn.rnn import ARMA, ExtendedLSTM

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


def test_ARMA():
    """Test nn.rnn.ARMA."""
    print(f"Testing ARMA with [{device}].")
    torch.enable_grad()
    G = ARMA(1, 30, 30, [10]*3).to(device)
    ins = torch.rand(10, 50, 1, requires_grad=True).to(device)
    out = G.filter(ins)
    assert ins.shape == out.shape


def test_LSTM():
    """Test nn.rnn.ExtendedLSTM."""
    print(f"Testing ExtendedLSTM with [{device}].")
    net = ExtendedLSTM(18, 24)
    x = torch.randn(100, 16, 18)
    y, (hy, cy) = net(x)
    return


if __name__ == "__main__":
    test_ARMA()
    test_LSTM()
