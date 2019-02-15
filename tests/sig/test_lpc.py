"""Test suite for LPC."""
import numpy as np


def test_lpc():
    """Test LPC using 3 methods."""
    from audlib.io.audio import audioread
    from audlib.sig.temporal import lpc_atc, lpc_cov, lpc_parcor, ref2pred
    sig, sr = audioread('samples/welcome16k.wav')
    frame = sig[16999:17319]
    order = 14
    alphas_atc, gain_atc = lpc_atc(frame, order)
    alphas_cov, gain_cov = lpc_cov(frame, order)
    ks_par, gain_par = lpc_parcor(frame, order)
    alphas_par = ref2pred(ks_par)

    # Covariance method is different than the others in this case
    assert np.allclose(alphas_atc, alphas_par)
    assert np.allclose(gain_atc, gain_par)


if __name__ == '__main__':
    test_lpc()
