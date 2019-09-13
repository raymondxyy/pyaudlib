"""Test suite for RNNs."""
import torch
from torch.nn.utils.rnn import pack_sequence

from audlib.nn.rnn import ARMA, ExtendedLSTM, ExtendedGRU
from audlib.nn.util import UnpackedSequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')


def test_ARMA():
    """Test nn.rnn.ARMA."""
    print(f"Testing ARMA with [{device}].")
    torch.enable_grad()
    G = ARMA(1, 30, 30, [10]*3).to(device)
    ins = torch.rand(10, 50, 1, requires_grad=True).to(device)
    out = G.filter(ins)
    assert ins.shape == out.shape


def test_ExtendedLSTM():
    """Test nn.rnn.ExtendedLSTM."""
    print(f"Testing ExtendedLSTM with [{device}].")
    batchsize = 16
    indim, outdim = 18, 24
    nframes = 100

    for numlayer in (1, 2):
        net = ExtendedLSTM(indim, outdim, numlayer)

        # Test normal tensors
        x = torch.randn(nframes, batchsize, indim)
        y, (hn, cn) = net(x)
        assert y.shape == (nframes, batchsize, outdim)
        assert hn.shape == cn.shape == (numlayer, batchsize, outdim)

        # Test UnpackedSequence
        xps = pack_sequence(
            [torch.randn(nframes, indim) for i in range(batchsize)]
        )
        yps, (hn, cn) = net(xps)
        assert all(y.shape == (nframes, outdim) for y in UnpackedSequence(yps))
        assert hn.shape == cn.shape == (numlayer, batchsize, outdim)
    return


def test_ExtendedGRU():
    """Test nn.rnn.ExtendedGRU."""
    print(f"Testing ExtendedGRU with [{device}].")
    batchsize = 16
    indim, outdim = 18, 24
    nframes = 100

    for numlayer in (1, 2):
        net = ExtendedGRU(indim, outdim, numlayer)

        # Test normal tensors
        x = torch.randn(nframes, batchsize, indim)
        y, hn = net(x)
        assert y.shape == (nframes, batchsize, outdim)
        assert hn.shape == (numlayer, batchsize, outdim)

        # Test UnpackedSequence
        xps = pack_sequence(
            [torch.randn(nframes, indim) for i in range(batchsize)]
        )
        yps, hn = net(xps)
        assert all(y.shape == (nframes, outdim) for y in UnpackedSequence(yps))
        assert hn.shape == (numlayer, batchsize, outdim)
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
    test_UnpackedSequence()
    test_ARMA()
    test_ExtendedGRU()
    test_ExtendedLSTM()
