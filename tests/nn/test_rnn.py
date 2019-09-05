"""Test suite for RNNs."""
import torch
from torch.nn.utils.rnn import pack_sequence

from audlib.nn.rnn import ARMA, ExtendedLSTM, UnpackedSequence

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


def test_UnpackedSequence():
    """Test nn.rnn.UnpackedSequence."""
    print(f"Testing UnpackedSequence with [{device}].")
    seqs = [torch.rand(20 if i < 10 else 15, 256) for i in range(15)]
    ps = pack_sequence(seqs)
    for s1, s2 in zip(UnpackedSequence(ps), seqs):
        assert torch.allclose(s1, s2)

    return

if __name__ == "__main__":
    test_ARMA()
    test_LSTM()
    test_UnpackedSequence()
