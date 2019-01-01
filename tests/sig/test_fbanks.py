from audlib.sig.fbanks import MelFreq
from audlib.io.audio import audioread
from audlib.plot import cepline
import matplotlib.pyplot as plt
import numpy as np
sig, sr = audioread('samples/welcome16k.wav')
nstart = int(sr*0.534)  # Rich Stern uttering 'ee' of 'D'.
nfft = 512
dee = sig[nstart:nstart+nfft]
melbank = MelFreq(sr, nfft, 20)
ndct = 20
fig = plt.figure(figsize=(16, 12), dpi=100)
ax1 = fig.add_subplot(211)
mfcc_dee = melbank.mfcc(dee)[:ndct]
cepline(np.arange(ndct), mfcc_dee, ax1)
