"""Utilities for noise estimation."""
import numpy as np
from scipy.special import gammainc


def mmse_henriks(sigpspec, noise=None, alpha=.98, beta=.8, rule='wiener'):
    """Implement the MMSE method by Henriks et al for noise PSD estimation.

    Hendriks, Richard C., Richard Heusdens, and Jesper Jensen.
    "MMSE based noise PSD tracking with low complexity."
    2010 IEEE International Conference on Acoustics, Speech and Signal
    Processing. IEEE, 2010.
    Retrieved from http://cas.et.tudelft.nl/pubs/0004266.pdf

    Parameters
    ----------
    sigpspec: array_like
        Power spectra of the noisy speech to be analyzed.
    noise: array_like, optional
        Optional examplary noise signal from which noise PSD is extracted.
    alpha: float, optional
        Recursive averaging coefficient for decision-directed a priori SNR.
        Default to .98.
    beta: float, optional
        Recursive averaging coefficient for noise PSD.
        Default to .8, as suggested by Henriks et al.
    rule: str, optional
        A priori SNR-to-spectral gain rule. Default to 'wiener'.
        Options are: wiener, specsub, ml

    See Also
    --------
    priori2filt, enhance.mmse_henriks

    """
    # Estimate noise PSD first;  will be used as Pnn[-1]
    nframes = sigpspec.shape[0]
    if noise is None:
        npsd_init = sigpspec[:min(6, nframes), :].mean(axis=0)
    else:
        npsd_init = noise
    npsd = np.empty((nframes+1, len(npsd_init)))  # include Pnn[-1]
    priori = np.empty_like(npsd)
    posteri = np.empty_like(npsd)
    npsd[0] = npsd_init
    posteri[0] = 1
    priori[0] = alpha + (1-alpha)*np.maximum(posteri[0] - 1, 0)
    for i, pspec in enumerate(sigpspec):
        posteri[i+1] = pspec / npsd[i]
        priori_hat = np.maximum(posteri[i+1] - 1, 0)  # half-wave rectify
        npsd_hat = (1/(1+priori_hat)**2
                    + priori_hat/((1+priori_hat)*posteri[i+1])) * pspec
        priori[i+1] = alpha*(priori2filt(priori[i], rule)**2) *\
            posteri[i] + (1-alpha)*priori_hat
        bias = 1 / ((1+priori[i+1]) * gammainc(2, 1/(1+priori[i+1]))
                    + np.exp(-1/(1+priori[i+1])))
        npsd_hat *= bias  # apply correction term to Pnn

        npsd[i+1] = beta * npsd[i] + (1-beta) * npsd_hat
        if i == sigpspec.shape[0]:
            break

    return npsd[1:], priori[1:], posteri[1:]


def priori2filt(priori, rule='wiener'):
    """Compute filter gains given a priori SNR."""
    if rule == 'wiener':
        return priori / (1+priori)
    elif rule == 'specsub':
        return np.sqrt(priori / (1+priori))
    elif rule == 'ml':
        return 0.5*(1 + np.sqrt(priori / (1+priori)))
    else:
        raise NotImplementedError(
            "Suppression rules must be [wiener/specsub/ml]")
