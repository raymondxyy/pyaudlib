from audlib.sig.fbanks import MelFreq
from audlib.io.audio import audioread
from audlib.plot import cepline
from audlib.sig.window import hamming
from audlib.sig.transform import stmfcc
import matplotlib.pyplot as plt
import numpy as np

sig, sr = audioread('samples/welcome16k.wav')


def plot_dee():
    """Plot cepstrum of Rich Stern uttering 'ee' of D."""
    nstart = int(sr*0.534)
    nfft = 512
    dee = sig[nstart:nstart+nfft]
    melbank = MelFreq(sr, nfft, 20)
    ndct = 20
    fig = plt.figure(figsize=(16, 12), dpi=100)
    ax1 = fig.add_subplot(211)
    mfcc_dee = melbank.mfcc(dee)[:ndct]
    cepline(np.arange(ndct), mfcc_dee, ax1)
    plt.show()


def test_mfcc():
    # TODO: need to add proper testing.
    nfft = 512
    nmel = 40
    melbank = MelFreq(sr, nfft, nmel)
    window_length = 0.032
    wind = hamming(int(window_length*sr))
    hop = .25
    mfcc = stmfcc(sig, sr, wind, hop, nfft, melbank)
    return mfcc


if __name__ == '__main__':
    test_mfcc()
