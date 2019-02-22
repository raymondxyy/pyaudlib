"""Test suites for pitch estimators."""
import os

import numpy as np
from scipy.signal import chirp

from audlib.io.audio import audioread
from audlib.pitch import HistoPitch
from audlib.sig.window import rect
from audlib.sig.fbanks import Gammatone

TEST_SIGNAL = 'speech'

if TEST_SIGNAL == 'speech':
    sigpath = os.getenv('speech')
    if not sigpath:
        print("Define env variable `speech=/path/to/speech.wav`! Exit.")
        exit()
    sig, sr = audioread(sigpath)
    t = np.arange(len(sig))/sr
elif TEST_SIGNAL == 'chirp':
    sr = 16000
    t = np.arange(sr*2) * (1./sr)
    sig = chirp(t, f0=200, f1=400, t1=2, method='linear')
elif TEST_SIGNAL == 'tone':
    sr = 16000
    t = np.arange(sr*2) * (1./sr)
    sig = np.sin(2*np.pi*200*t)

# Same configuration as in Mike Seltzer's paper
window_length = 0.025
hop_length = 0.010
hop = int(hop_length*sr)
wind = rect(int(window_length*sr), hop=hop)
nfft = 512

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    hist_based = HistoPitch(
        Gammatone(sr, 40, center_frequencies=(150, 6000)), sr, wind, hop, lpc_order=0)
    hist = hist_based.t0hist(sig)
    fig = plt.figure()
    ax0 = fig.add_subplot(611)
    ax0.plot(t, sig)
    ax0.set_ylabel("Waveform")

    # Plot raw pitch estimate
    tt = np.arange(len(hist))*hop_length
    # Plot voice decision and pitch contour
    uv1, pitch1 = hist_based.pitchcontour(hist, neighbor=.1)
    rawpitch = [sr/np.argmax(f) if uv1[ii] else 0 for ii, f in enumerate(hist)]
    ax1 = fig.add_subplot(612)
    ax1.plot(tt, rawpitch)
    ax1.set_ylabel("Raw pitch")
    for ii, (ll, pp) in enumerate(zip(uv1, pitch1)):
        if ll:
            print(f"{pp:.2f}")
        else:
            print("0.00")

    pitch2 = hist_based.pitchcontour2(hist, uv1, neighbor=.1)

    ax2 = fig.add_subplot(613)
    ax2.plot(tt, uv1)
    ax2.set_ylabel("Voiced/Unvoiced")
    ax3 = fig.add_subplot(614)
    ax3.plot(tt, pitch1)
    ax3.set_ylabel("Histopitch-1")
    ax4 = fig.add_subplot(615)
    ax4.plot(tt, pitch2)
    ax4.set_ylabel("Histopitch-2")
    ax5 = fig.add_subplot(616)
    ax5.plot(tt, hist_based.runningvar(uv1, pitch2))
    ax5.set_ylabel("H2 Running-Var")
    ax5.set_xlabel("Time (s)")
    fig2 = plt.figure()
    ax = fig2.add_subplot(111)
    from audlib.plot import specgram
    specgram(hist, ax)
    ax.set_title("T0 Histogram")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Lag")
    plt.show()
