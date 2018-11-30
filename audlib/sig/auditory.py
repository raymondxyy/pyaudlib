"""Auditory Filterbanks and scales for Speech and Audio Analysis.

The Gammatone filterbank is a direct translation of Dan Ellis' Gammatone-like
spectrograms package [1], which is partly and a direct translation of Malcolm
Slaney's Auditory toolbox [2].

References:
   [1]: https://labrosa.ee.columbia.edu/matlab/gammatonegram/
   [2]: https://engineering.purdue.edu/~malcolm/interval/1998-010/

"""

import numpy as np
from scipy import signal


def hz2mel(f, sphinx=True):
    """Convert linear frequency to mel frequency scale."""
    if sphinx:
        return 2595. * np.log10(1+f/700.)
    # match Slaney's toolbox
    f0, f_sp, brkfrq = 0., 200./3, 1000.
    brkpt = (brkfrq - f0) / f_sp
    logstep = np.exp(np.log(6.4)/27.)

    z = np.empty_like(f)
    lower = f < brkfrq  # np.less(f,brkfrq)
    higher = np.logical_not(lower)

    z[lower] = (f[lower] - f0) / f_sp
    z[higher] = brkpt + np.log(f[higher]/brkfrq) / np.log(logstep)
    return z


def mel2hz(z, sphinx=True):
    """Convert Mel frequency to linear frequency scale."""
    if sphinx:
        return 700*(10**(z/2595.)-1)

    f0, f_sp, brkfrq = 0., 200./3, 1000.
    brkpt = (brkfrq - f0) / f_sp
    logstep = np.exp(np.log(6.4)/27.)

    f = np.empty_like(z)
    lower = z < brkpt  # np.less(z,brkpt)
    higher = np.logical_not(lower)

    f[lower] = f0 + z[lower] * f_sp
    f[higher] = brkfrq * np.exp(np.log(logstep)*(z[higher]-brkpt))
    return f

# ERB-related Functions starting below


# Global Parameters
# Change the following three parameters if you wish to use a different
# ERB scale.  Must change in MakeERBCoeffs too.
ERB_EAR_Q = 9.26449  # Glasberg and Moore Parameters
ERB_MIN_BW = 24.7
ERB_ORDER = 1


# Process an input waveform with a gammatone filter bank. This function
# takes a single sound vector, and returns an array of filter outputs, one
# channel per row.
#
# The fcoefs parameter, which completely specifies the Gammatone filterbank,
# should be designed with the MakeERBFilters function.  If it is omitted,
# the filter coefficients are computed for you assuming a 22050Hz sampling
# rate and 64 filters regularly spaced on an ERB scale from fs/2 down to 100Hz.
#
# Malcolm Slaney @ Interval, June 11, 1998.
# (c) 1998 Interval Research Corporation
# Thanks to Alain de Cheveigne' for his suggestions and improvements.
def erb_fbank(sig, fcoefs, band=None):
    """Filter a signal using ERB filterbanks."""
    A0, A11, A12, A13, A14, A2, B0, B1, B2, gain = fcoefs
    if band is not None:  # select bands
        A0 = A0[band]
        A11 = A11[band]
        A12 = A12[band]
        A13 = A13[band]
        A14 = A14[band]
        A2 = A2[band]
        B0 = B0[band]
        B1 = B1[band]
        B2 = B2[band]
        gain = gain[band]

    y = np.zeros((gain.shape[0], len(sig)))
    for i, chan in enumerate(range(gain.shape[0]-1, -1, -1)):
        # loop through each subband from high to low
        y1 = signal.lfilter([A0[chan]/gain[chan],
                             A11[chan]/gain[chan],
                             A2[chan]/gain[chan]],
                            [B0[chan], B1[chan], B2[chan]], sig)
        y2 = signal.lfilter([A0[chan], A12[chan], A2[chan]],
                            [B0[chan], B1[chan], B2[chan]], y1)
        y3 = signal.lfilter([A0[chan], A13[chan], A2[chan]],
                            [B0[chan], B1[chan], B2[chan]], y2)
        y[i, :] = signal.lfilter([A0[chan], A14[chan], A2[chan]],
                                 [B0[chan], B1[chan], B2[chan]], y3)
    return y


# Directly copy from Ellis' package. Below is his description:
# This function computes the filter coefficients for a bank of
# Gammatone filters.  These filters were defined by Patterson and
# Holdworth for simulating the cochlea.
#
# The result is returned as an array of filter coefficients.  Each row
# of the filter arrays contains the coefficients for four second order
# filters.  The transfer function for these four filters share the same
# denominator (poles) but have different numerators (zeros).  All of these
# coefficients are assembled into one vector that the **ERBFilterBank**
# can take apart to implement the filter.
#
# The filter bank contains **num_chan** channels that extend from
# half the sampling rate (fs) to **low_freq**.  Alternatively, if the num_chan
# input argument is a vector, then the values of this vector are taken to
# be the center frequency of each desired filter.  (The low_freq argument is
# ignored in this case.)
#
# Note this implementation fixes a problem in the original code by
# computing four separate second order filters.  This avoids a big
# problem with round off errors in cases of very small cfs (100Hz) and
# large sample rates (44kHz).  The problem is caused by roundoff error
# when a number of poles are combined, all very close to the unit
# circle.  Small errors in the eigth order coefficient, are multiplied
# when the eigth root is taken to give the pole location.  These small
# errors lead to poles outside the unit circle and instability.  Thanks
# to Julius Smith for leading me to the proper explanation.
def erb_filters(sr, num_chan, low_freq):
    """Construct ERB filterbanks."""
    T = 1./sr
    if type(num_chan) == int:
        # make center frequencies in this case
        cf = erb_space(low_freq, int(sr/2), num_chan)
    else:
        cf = num_chan.copy()
    ERB = ((cf/ERB_EAR_Q)**ERB_ORDER + ERB_MIN_BW**ERB_ORDER)**(1/ERB_ORDER)
    B = 1.019*2*np.pi*ERB
    A0 = T
    A2 = 0.
    B0 = 1.
    B1 = -2*np.cos(2*cf*np.pi*T) / np.exp(B*T)
    B2 = np.exp(-2*B*T)

    A11 = -(2*T*np.cos(2*cf*np.pi*T)/np.exp(B*T) +
            2*np.sqrt(3+2**1.5)*T*np.sin(2*cf*np.pi*T) / np.exp(B*T))/2
    A12 = -(2*T*np.cos(2*cf*np.pi*T)/np.exp(B*T) -
            2*np.sqrt(3+2**1.5)*T*np.sin(2*cf*np.pi*T) / np.exp(B*T))/2
    A13 = -(2*T*np.cos(2*cf*np.pi*T)/np.exp(B*T) +
            2*np.sqrt(3-2**1.5)*T*np.sin(2*cf*np.pi*T) / np.exp(B*T))/2
    A14 = -(2*T*np.cos(2*cf*np.pi*T)/np.exp(B*T) -
            2*np.sqrt(3-2**1.5)*T*np.sin(2*cf*np.pi*T) / np.exp(B*T))/2

    gain = np.abs(
        (-2*np.exp(4*1j*cf*np.pi*T)*T +
         2*np.exp(-(B*T) + 2*1j*cf*np.pi*T)*T *
         (np.cos(2*cf*np.pi*T) - np.sqrt(3 - 2**(3./2)) *
          np.sin(2*cf*np.pi*T))) *
        (-2*np.exp(4*1j*cf*np.pi*T)*T +
         2*np.exp(-(B*T) + 2*1j*cf*np.pi*T)*T *
         (np.cos(2*cf*np.pi*T) + np.sqrt(3 - 2**(3./2)) *
          np.sin(2*cf*np.pi*T))) *
        (-2*np.exp(4*1j*cf*np.pi*T)*T +
         2*np.exp(-(B*T) + 2*1j*cf*np.pi*T)*T *
         (np.cos(2*cf*np.pi*T) -
          np.sqrt(3 + 2**(3./2))*np.sin(2*cf*np.pi*T))) *
        (-2*np.exp(4*1j*cf*np.pi*T)*T+2*np.exp(-(B*T)+2*1j*cf*np.pi*T)*T *
         (np.cos(2*cf*np.pi*T) + np.sqrt(3+2**(3./2))*np.sin(2*cf*np.pi*T))) /
        (-2 / np.exp(2*B*T) - 2*np.exp(4*1j*cf*np.pi*T) +
         2*(1 + np.exp(4*1j*cf*np.pi*T))/np.exp(B*T))**4)

    allfilts = np.ones((len(cf)))
    return (A0*allfilts, A11, A12, A13, A14,
            A2*allfilts, B0*allfilts, B1, B2, gain), cf


def erb_space(low_freq=100., high_freq=11025., N=100):
    """Construct linear frequencies on an ERB scale."""
    # This function computes an array of N frequencies uniformly spaced between
    # highFreq and lowFreq on an ERB scale.  N is set to 100 if not specified.
    #
    # See also linspace, logspace, MakeERBCoeffs, MakeERBFilters.
    #
    # For a definition of ERB, see Moore, B. C. J., and Glasberg, B. R. (1983).
    # "Suggested formulae for calculating auditory-filter bandwidths and
    # excitation patterns," J. Acoust. Soc. Am. 74, 750-753.
    q_times_bw = ERB_EAR_Q*ERB_MIN_BW
    return -q_times_bw+np.exp((np.arange(N)+1)*(
        -np.log(high_freq+q_times_bw) + np.log(
            low_freq + q_times_bw))/N) * (high_freq + q_times_bw)
