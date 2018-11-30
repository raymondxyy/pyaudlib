# Utility functions related to audio processing
# Author: Raymond Xia

import numpy as np
import numpy.random as rand

def sample(x, length, num, verbose=False):
    """
    Given audio x, sample `num` segments with `length` samples each
    """
    assert len(x) >= length
    segs = []
    start_idx_max = len(x)-length
    start_idx = np.around(rand.rand(num) * start_idx_max)
    for i in start_idx:
        segs.append(x[int(i):int(i)+length])
        if verbose:
            print('Take samples {} to {}...'.format(str(i),str(i+length)))
    return segs

def sample_pair(x,y,length,num,verbose=False):
    """
    Assume y has same dimension as x.
    """
    maxlength = min(len(x),len(y))
    assert maxlength >= length
    xsegs,ysegs = [],[]
    start_idx_max = maxlength-length
    start_idx = np.around(rand.rand(num) * start_idx_max)
    for i in start_idx:
        xsegs.append(x[int(i):int(i)+length])
        ysegs.append(y[int(i):int(i)+length])
        if verbose:
            print('Take samples {} to {}...'.format(str(i),str(i+length)))
    return xsegs,ysegs

def add_noise(x,n,snr=None):
    """
    Add user provided noise n with SNR=snr to signal x.
    SNR = 10log10(Signal Energy/Noise Energy)
    NE = SE/10**(SNR/10)
    """

    # Take care of size difference in case x and n have different shapes
    xlen,nlen = len(x),len(n)
    if xlen > nlen: # need to append noise several times to cover x range
        nn = np.tile(n,xlen/nlen+1)
        nlen = len(nn)
    else:
        nn = n
    if xlen < nlen: # slice a portion of noise
        nn = sample(nn,xlen,1)[0]
    else: # equal length
        nn = nn

    if snr is None: snr = (rand.random()-0.25)*20
    xe = x.dot(x) # signal energy
    ne = nn.dot(nn) # noise power
    nscale = np.sqrt(xe/(10**(snr/10.)) /ne) # scaling factor
    return x + nscale*nn

def add_white_noise(x,snr=None):
    """
    Add white noise with SNR=snr to signal x.
    SNR = 10log10(Signal Energy/Noise Energy) = 10log10(SE/var(noise))
    var(noise) = SE/10**(SNR/10)
    """
    n = rand.normal(0,1,x.shape)
    return add_noise(x,n,snr)

def white_noise(x,snr=None):
    """
    Instead of adding white noise to input x, simply return the white noise
    array.
    """
    n = rand.normal(0,1,x.shape)
    if snr is None: snr = (rand.random()-0.25)*20
    xe = x.dot(x) # signal energy
    ne = n.dot(n) # noise power
    nscale = np.sqrt(xe/(10**(snr/10.)) /ne) # scaling factor
    return nscale*n


def normalize(x):
    """
    Normalize signal amplitude to be in range [-1,1]
    """
    return x/np.max(np.abs(x))


add_white_noise0db = lambda x: add_white_noise(x,0.)
add_white_noise5db = lambda x: add_white_noise(x,5.)
add_white_noise15db = lambda x: add_white_noise(x,15.)
add_white_noise25db = lambda x: add_white_noise(x,25.)

# Add white noise with SNR in range [-10dB,10dB]
add_white_noise_rand = lambda x: add_white_noise(x,(rand.random()-0.25)*20)


def quantize(x,n):
    x /= np.ma.max(np.abs(x)) # make sure x in [-1,1]
    bins = np.linspace(-1,1,2**n+1,endpoint=True) # [-1,1]
    qvals = (bins[:-1] + bins[1:]) / 2
    bins[-1] = 1.01 # Include 1 in case of clipping
    return qvals[np.digitize(x,bins)-1]

quantize_1bit = lambda x: quantize(x,1)
quantize_4bit = lambda x: quantize(x,4)
quantize_8bit = lambda x: quantize(x,8)
quantize_16bit = lambda x: quantize(x,16)
