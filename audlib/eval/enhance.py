"""Evaluation metrics for speech enhancement tasks."""
import numpy as np


def sisdr(estim, target, sdr_min=-np.inf, sdr_max=np.inf):
    """Scale-invariant signal-tp-distortion ratio.

    Original paper: https://arxiv.org/abs/1811.02508.

    Parameters
    ----------
    estim: array_like
        estimimated signal.
        Shape can be either (T,) or (B, T), where B is the batch size.
    target: array_like
        targetget signal.
        Shape must be (T,).

    Returns
    -------
    sdr: float or list of float
        Signal-to-distortion ratio.

    """
    assert estim.shape[-1] == len(target), "signals must have equal lengths."

    def _snr(te, de):
        if te == 0:
            return sdr_min
        elif de == 0:
            return sdr_max
        else:
            return 10*(np.log10(te) - np.log10(de))

    te = target.dot(target)
    if len(estim.shape) == 1:
        target = estim.dot(target) * target / te
        disto = estim - target
        return _snr(target.dot(target), disto.dot(disto))
    else:  # batch mode
        target = np.outer(estim.dot(target), target) / te
        disto = estim - target
        tes = (tt.dot(tt) for tt in target)
        des = (dd.dot(dd) for dd in disto)
        return [_snr(te, de) for te, de in zip(tes, des)]
