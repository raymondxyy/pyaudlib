"""Test suite for RNNs."""
import torch

from audlib.nn.rnn import ARMA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_arma():
    """Test nn.rnn.ARMA."""
    print(f"Testing ARMA with [{device}].")
    G = ARMA(1, 30, 30).to(device)
    ins = torch.rand(10, 5, 1).to(device)
    out = G.filter(ins)
    assert ins.shape == out.shape

    target = torch.rand_like(out)
    loss = (target - out).mean()
    loss.backward()


if __name__ == "__main__":
    test_arma()
