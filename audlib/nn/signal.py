"""SIGNAL transform functions done in torch."""
import torch
import torch.nn.functional as F


def firfreqz(h, ndft, squared=False):
    """Compute frequency response of an FIR filter."""
    assert ndft > h.size(-1), "Incompatible DFT size!"
    h = F.pad(h, (0, ndft-h.size(-1)))
    hspec = torch.rfft(h, 1)
    hspec = hspec[..., 0]**2 + hspec[..., 1]**2
    if squared:
        return hspec
    return hspec**.5


def iirfreqz(h, ndft, squared=False, powerfloor=10**-3):
    """Compute frequency response of an IIR filter."""
    assert ndft > h.size(-1), "Incompatible DFT size!"
    h = F.pad(h, (0, ndft-h.size(-1)))
    hspec = torch.rfft(h, 1)
    hspec = (hspec[..., 0]**2 + hspec[..., 1]**2).clamp(min=powerfloor)
    if squared:
        return 1 / hspec
    return 1 / (hspec**.5)


def freqz(b, a, ndft, gain=1, iirfloor=10**-3, squared=False):
    """Compute the frequency response of a filter."""
    hhnum = firfreqz(b, ndft, squared)
    hhden = iirfreqz(a, ndft, squared, iirfloor)
    if squared:
        return hhnum * hhden * gain**2
    return hhnum*hhden * gain



def hilbert(x, ndft=None):
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
    if ndft is None:
        sig = x
    else:
        assert ndft > x.size(-1)
        sig = F.pad(x, (0, ndft-x.size(-1)))
    xspec = torch.rfft(sig, 1, onesided=False)
    siglen = sig.size(-1)
    h = torch.zeros(siglen, 2, dtype=sig.dtype, device=sig.device)
    if siglen % 2 == 0:
        h[0] = h[siglen//2] = 1
        h[1:siglen//2] = 2
    else:
        h[0] = 1
        h[1:(siglen+1)//2] = 2

    return torch.ifft(xspec * h, 1)
