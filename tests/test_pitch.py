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
        Gammatone(sr, 40, center_frequencies=(150, 6000)), sr, wind, hop)
    hist = hist_based.t0hist(sig)
    pest = [sr/np.argmax(frame) for frame in hist]
    fig = plt.figure()
    ax0 = fig.add_subplot(411)
    ax0.plot(t, sig)
    ax1 = fig.add_subplot(412)
    # Plot raw pitch estimate
    tt = np.arange(len(hist))*(hop*window_length)
    ax1.plot(tt, pest)
    # Plot voice decision and pitch contour
    vp = hist_based.pitchcontour(sig)
    for ii, (ll, pp) in enumerate(vp):
        if ll:
            print(f"{pp:.2f}")
        else:
            print("0.00")
        #if ll:
        #    time = ii * hop_length
        #    print(f"Time: [{time:.2f}s], Pitch: [{pp:.2f}Hz]")
    ax2 = fig.add_subplot(413)
    ax2.plot(tt, [ll for (ll, _) in vp])
    ax3 = fig.add_subplot(414)
    ax3.plot(tt, [pp for (_, pp) in vp])
    fig2 = plt.figure()
    ax = fig2.add_subplot(111)
    from audlib.plot import specgram
    specgram(hist, ax)
    plt.show()
