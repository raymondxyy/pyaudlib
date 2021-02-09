"""Test suites for pitch estimators."""

# TODO: Make test cases
import os

import numpy as np
from scipy.signal import chirp

from audlib.pitch import HistoPitch
from audlib.sig.window import rect
from audlib.sig.fbanks import Gammatone

TIMEIT = False

TEST_SIGNAL = 'speech'

if TEST_SIGNAL == 'chirp':
    sr = 16000
    t = np.arange(sr*2) * (1./sr)
    sig = chirp(t, f0=200, f1=400, t1=2, method='linear')
elif TEST_SIGNAL == 'tone':
    sr = 16000
    t = np.arange(sr*2) * (1./sr)
    sig = np.sin(2*np.pi*200*t)
elif TEST_SIGNAL == 'speech':
    path = os.getenv('SPEECH')
    if path is None:
        print("Specify SPEECH with export SPEECH=/path/to/speech.wav!")
        exit()
    from audlib.io.audio import audioread
    sig, sr = audioread(path)
    t = np.arange(len(sig)) / sr

# Same configuration as in Mike Seltzer's paper
window_length = 0.025
hop_length = 0.010
hop = int(hop_length*sr)
wind = rect(int(window_length*sr), hop=hop)
nfft = 512


def time_t0hist(method, sig, iters=10):
    from timeit import default_timer as timer
    start = timer()
    for ii in range(iters):
        method.t0hist(sig)
    end = timer()
    print("Elapsed time: {:.4f} per iteration after {} runs.".format(
            (end-start)/iters, iters))
    return

def test_histopitch():
    hist_based = HistoPitch(
        Gammatone(sr, 40, center_frequencies=(150, 6000)), sr, wind, hop,
        lpc_order=0)
    hist = hist_based.t0hist(sig)

    # Plot raw pitch estimate
    tt = np.arange(len(hist))*hop_length
    # Plot voice decision and pitch contour
    uv1, pitch1 = hist_based.pitchcontour(hist, neighbor=.1)
    rawpitch = [sr/np.argmax(f) if uv1[ii] else 0
                for ii, f in enumerate(hist)]
    for ii, (ll, pp) in enumerate(zip(uv1, pitch1)):
        if ll:
            print(f"{pp:.2f}")
        else:
            print("0.00")

    pitch2 = hist_based.pitchcontour2(hist, uv1, neighbor=.1)

def test_praatpitch():
    from audlib.pitch import PraatPitch

    praat = PraatPitch(sr, voiced_unvoiced_cost=0.4, max_pitch=400)
    pitches = praat(sig)
    from pprint import pprint
    pprint(pitches)


if __name__ == '__main__':
    test_praatpitch()
