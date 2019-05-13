"""SIGNAL transform functions done in torch."""
import torch


def hilbert(x):
    r"""Analytic signal of x.

    Return the analytic signal of a real signal x, x + j\hat{x}, where \hat{x}
    is the Hilbert transform of x.

    Parameters
    ----------
    x: torch.Tensor
        Audio signal to be analyzed.
        Always assumes x is real, and x.shape[-1] is the signal length.

    Returns
    -------
    out: torch.Tensor
        out.shape == (*x.shape, 2)

    """
    xspec = torch.rfft(x, 1, onesided=False)
    siglen = x.size(-1)
    h = torch.zeros(siglen, 2, dtype=x.dtype, device=x.device)
    if siglen % 2 == 0:
        h[0] = h[siglen//2] = 1
        h[1:siglen//2] = 2
    else:
        h[0] = 1
        h[1:(siglen+1)//2] = 2

    return torch.ifft(xspec * h, 1)
