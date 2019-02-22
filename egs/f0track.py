"""Pitch tracking using Audlib on the ARCTIC dataset."""
import os

import matplotlib.pyplot as plt
import numpy as np

from audlib.data.arctic import ARCTIC
from audlib.pitch import HistoPitch, ACF
from audlib.sig.window import rect
from audlib.sig.fbanks import Gammatone
from audlib.vad import ZCREnergy

# Configurations for pitch tracker: same as in Mike Seltzer's paper
SR = 16000
WINDOW_LENGTH = 0.025
HOP_LENGTH = 0.010
hop = int(HOP_LENGTH*SR)
wind = rect(int(WINDOW_LENGTH*SR), hop=hop)
nfft = 512

f0tracker1 = HistoPitch(
    Gammatone(SR, 40, center_frequencies=(150, 6000)),
    SR, wind, hop, lpc_order=14)
f0tracker2 = ACF(SR, wind, hop, lpc_order=14)

vad = ZCREnergy(SR, wind, hop)


def pitch_demo(sig):
    """Compute and plot the pitch contour of a signal."""
    hist = f0tracker1.t0hist(sig)
    uv1, pitch1 = f0tracker1.pitchcontour(hist)
    pitch2 = f0tracker1.pitchcontour2(hist, uv1, neighbor=.02)
    rawpitch = [SR/np.argmax(f) if uv1[ii] else 0 for ii, f in enumerate(hist)]

    # A series of plots
    fig = plt.figure()

    # (1) Waveform
    ax = fig.add_subplot(811)
    ax.plot(np.arange(len(sig))/SR, sig)
    ax.set_ylabel("Waveform")

    # (2) Raw pitch
    ax = fig.add_subplot(812)
    tt = np.arange(len(hist))*HOP_LENGTH
    ax.plot(tt, rawpitch)
    ax.set_ylabel("Raw pitch")

    # (3) Voiced/unvoiced decision
    ax = fig.add_subplot(813)
    ax.plot(tt, uv1)
    ax.set_ylabel("Voiced/Unvoiced")

    # (3.5) zero-crossing energy ratio
    ax = fig.add_subplot(814)
    ax.plot(tt, [(z, e) for (z, e) in vad.zcre(sig, lpc_order=14)])
    ax.set_ylabel("ZCRE")

    # (4) Pitch contour using Seltzer's smoothing method
    ax = fig.add_subplot(815)
    ax.plot(tt, pitch1)
    ax.set_ylabel("Histopitch-1")

    # (5) Pitch contour using Xia's smoothing method
    ax = fig.add_subplot(816)
    ax.plot(tt, pitch2)
    ax.set_ylabel("Histopitch-2")

    # (6) Pitch contour with ACF
    ax = fig.add_subplot(817)
    ax.plot(tt, [f0 if uv else 0 for (uv, f0) in zip(
        uv1, f0tracker2.pitchcontour(sig))])
    ax.set_ylabel("ACF")
    ax.set_xlabel("Time (s)")

    # (6) Running standard deviation of pitch change within a voiced segment
    ax = fig.add_subplot(818)
    ax.plot(tt, f0tracker1.runningvar(uv1, pitch2))
    ax.set_ylabel("H2 Running-Std")
    ax.set_xlabel("Time (s)")

    # Plot raw T0-histogram
    fig2 = plt.figure()
    ax = fig2.add_subplot(111)
    from audlib.plot import specgram
    specgram(hist, ax)
    ax.set_title("T0 Histogram")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Lag")


if __name__ == '__main__':
    ARCTIC_ROOT = os.getenv('ARCTIC')
    if not ARCTIC_ROOT:
        print("""
            This script **requires** the CMU_ARCTIC database.
            Set env with export ARCTIC=/path/to/database to use.
            """)
        exit()
    arctic = ARCTIC(ARCTIC_ROOT, sr=SR, egg=True)
    print(arctic)

    # Select a sample for demo
    print(f"Processing [{arctic.all_files[0]}]")
    for sig in (arctic[0]['data'], arctic[0]['egg']):
        pitch_demo(sig)

    plt.show()
