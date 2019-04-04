"""Test suite for speech enhancement metrics."""
import numpy as np

from audlib.quickstart import welcome
from audlib.sig.util import add_white_noise
from audlib.eval.enhance import sisdr


def test_sisdr():
    """Test scale-invariant signal-to-distortion ratio."""
    sig, _ = welcome()
    print("Testing single mode with speech plus white noise...")
    snr = 10
    est = add_white_noise(sig, snr=snr)
    sdr = sisdr(est, sig)
    print(f"\tSNR: [{snr:.2f} dB], SDR: [{sdr:.2f} dB]")
    print("Testing batch mode with speech plus white noise...")
    snrs = [s for s in range(-20, 21)]
    est = np.empty((len(snrs), len(sig)))
    for ii, snr in enumerate(snrs):
        est[ii] = add_white_noise(sig, snr=snr)
    sdrs = sisdr(est, sig)
    for snr, sdr in zip(snrs, sdrs):
        print(f"\tSNR: [{snr:.2f} dB], SDR: [{sdr:.2f} dB]")


if __name__ == "__main__":
    test_sisdr()
