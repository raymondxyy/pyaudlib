"""Speech enhancement utilities."""
# Change log:
#   09/07/17:
#       * Create this file
#       * Added iterative Wiener filtering
#   11/26/17:
#       * Added A-Priori SNR estimation for Wiener filtering
#   01/01/19:
#       * Cleaned up to conform to new interface

import numpy as np
from numpy.fft import rfft, irfft
from scipy.signal import lfilter

from .sig.stproc import stana, ola
from .sig.fbanks import Gammatone
from .sig.temporal import lpc_parcor, ref2pred
from .sig.transform import stft, istft, stpsd
from .sig.spectemp import ssf, invspec
from .noise import mmse_henriks as npsd_henriks


class SSFEnhancer(object):
    """Suppression of Slowly-varying components and the Falling edge.

    This implementation follows paper by Kim and Stern:
    Kim, Chanwoo, and Richard M. Stern."Nonlinear enhancement of onset
    for robust speech recognition." Eleventh Annual Conference of the
    International Speech Communication Association. 2010.

    See Also
    --------
    spectemp.ssf

    """
    def __init__(self, sr, wind, hop, nfft, num_chan=40):
        """Instantiate an SSF enhancer.

        Parameters
        ----------
        sr: int
            Sampling rate in Hz.
        wind: numpy.ndarray
            Window function.
        hop: float
            Hop fraction.
        nfft: int
            Number of DFT points.

        Keyword Parameters
        ------------------
        num_chan: int, 40
            Number of channels in the Gammatone filterbank.
        ptype: int, 2
            SSF type. Default to 2.

        """
        assert sr > 9000, "Sampling rate too low!"
        self.gbank = Gammatone(sr, num_chan, (130., 4500.))
        self.gammawgt = self.gbank.gammawgt(nfft)

        def _stft(sig):
            return stft(sig, wind, hop, nfft, synth=True, zphase=True)

        def _istft(spec):
            return istft(spec, wind, hop, nfft, zphase=True)

        self.stft = _stft
        self.istft = _istft


    def __call__(self, sig, lambda_lp, c0=.01, ptype=2, pre_emphasis=True):
        """Enhance a signal with SSF."""
        if pre_emphasis:
            sig = lfilter([1, -.97], [1], sig)

        sigspec = self.stft(sig)
        gpmask = ssf((sigspec.real**2 + sigspec.imag**2) @ self.gammawgt,
                     lambda_lp, c0, ptype)
        pmask = invspec(gpmask, self.gammawgt)
        out = self.istft(np.sqrt(pmask) * sigspec)

        if pre_emphasis:
            out = lfilter([1], [1, -.97], out)

        return out


def wiener_iter(x, sr, wind, hop, nfft, noise=None, zphase=True, iters=3,
                lpc_order=12):
    # TODO: convert this to a class
    """Implement iterative Wiener filtering described by Lim and Oppenheim.

    Code based on Loizou's matlab implementation. See transform.stft for
    complete explanation of parameters.

    Parameters
    ----------
    noise: array_like
        Noise signal as a ndarray.
    iters: int
        Number of iterations to do between spectra and LPCs.
        Default to 3, as suggested in Loizou.

    See Also
    --------
    transform.stft

    """
    # Form the constant complex exponential matrix here for efficiency
    _exp_mat = np.exp(-1j*(2*np.pi)/nfft*np.outer(np.arange(
        lpc_order+1), np.arange(nfft//2+1)))

    # Basically copy from stft code to prevent redundant processing
    fsize = len(wind)
    woff = (fsize-(fsize % 2)) // 2
    zp = np.zeros(nfft-fsize)  # zero padding

    # Estimate noise PSD first
    if noise is None:
        npsd = stpsd(x, wind, hop, nfft, nframes=6)
    else:
        npsd = stpsd(noise, wind, hop, nfft, nframes=-1)

    def wiener_iter_frame(frame):
        """One frame of Wiener iterative filtering."""
        ref, _ = lpc_parcor(frame, lpc_order)
        xlpc = ref2pred(ref)
        # Get initial PSD estimate
        if zphase:
            frame = np.concatenate((frame[woff:], zp, frame[:woff]))
        else:  # conventional linear-phase STFT
            frame = np.concatenate((frame, zp))
        xspec = rfft(frame)
        xpsd = np.abs(xspec)**2

        # Iterative process here
        for ii in range(iters):  # iteration number influences spectral peaks
            # estimated psd of speech
            spsd = 1 / (np.abs(xlpc.dot(_exp_mat))**2)
            gain = max(np.mean(xpsd-npsd) / np.mean(spsd), 1e-16)
            spsd *= gain  # estimated speech PSD
            filt = spsd / (spsd + npsd)  # Wiener filter
            xspec = xspec * filt  # estimate new S(w) from Wiener filter
            # update LPC coefficients
            buff = irfft(xspec)
            if zphase:
                buff = np.roll(buff, nfft//2)
            ref, _ = lpc_parcor(buff, lpc_order)
            xlpc = ref2pred(ref)
        return buff, filt, spsd

    # Now perform frame-level Wiener filtering on x
    srec = []
    for xframe in stana(x, wind, hop, synth=True):
        xclean, _, _ = wiener_iter_frame(xframe)
        srec.append(xclean)

    # Synthesis by overlap-add (OLA)
    xsynth = ola(srec, wind, hop)

    return xsynth


def asnr(x, sr, wind, hop, nfft, noise=None, zphase=True, snrsmooth=.98,
         noisesmooth=.98, llkthres=.15, rule='wiener'):
    # TODO: convert this to a class
    """Implement the a-priori SNR estimation described by Scalart and Filho.

    Code based on Loizou's matlab implementation. Three suppression rules are
    available:
        1. Wiener (default)
        2. Spectral subtraction
        3. Maximum Likelihood

    Parameters
    ----------
    rule: str
        Filtering rule to apply given a priori SNR.
        Default to 'wiener'. Other options are 'specsub' and 'ml'.

    See Also
    --------
    wiener_iter, priori2filt

    """
    # Estimate noise PSD first
    if noise is None:
        npsd = stpsd(x, wind, hop, nfft, nframes=6)
    else:
        npsd = stpsd(noise, wind, hop, nfft, nframes=-1)

    xfilt = []  # holds Wiener-filtered output
    vad = []  # holds voice activity decision
    priori_m1 = np.zeros_like(npsd)
    posteri_m1 = np.zeros_like(npsd)
    for i, xspec in enumerate(stft(x, wind, hop, nfft, synth=True,
                                   zphase=zphase)):
        xpsd = np.abs(xspec)**2
        posteri = xpsd / npsd
        posteri_prime = np.maximum(posteri - 1, 0)  # half-wave rectify
        if i == 0:  # initialize priori SNR
            priori = snrsmooth + (1-snrsmooth)*posteri_prime
        else:
            priori = snrsmooth*(priori2filt(priori_m1, rule)**2) *\
                posteri_m1 + (1-snrsmooth)*posteri_prime
        # compute speech presence log likelihood
        llk = posteri*priori/(1+priori) - np.log(1+priori)
        vad.append(np.mean(llk))
        if vad[i] < llkthres:
            # noise only frame found, update Pn
            npsd = noisesmooth*npsd + (1-noisesmooth)*xpsd

        filt = priori2filt(priori, rule=rule)  # compute gain
        # filter and store cleaned frame
        xfilt.append(filt*xspec)

        # update old values
        priori_m1[:] = priori
        posteri_m1[:] = posteri

    # Synthesis by overlap-add (OLA)
    xout = istft(xfilt, wind, hop, nfft, zphase=zphase)

    return xout, vad


def asnr_spec(noisyspec, snrsmooth=.98, noisesmooth=.98, llkthres=.15,
              rule='wiener'):
    """Implement the a-priori SNR estimation described by Scalart and Filho.

    This is very similar to `asnr`, except it's computed directly on noisy
    magnitude spectra instead of time-domain signals.

    Outputs
    -------
    xfilt: ndarray
        Filtered magnitude spectrogram.
    priori: ndarray
        A priori SNR.
    posteri: ndarray
        A posteriori SNR.
    vad: ndarray
        Speech-presence log likelihood ratio.

    """
    # Estimate noise PSD first
    npsd = np.mean(noisyspec[:6, :]**2, axis=0)
    xfilt = np.zeros_like(noisyspec)  # holds Wiener-filtered output
    vad = []  # holds voice activity decision
    priori = np.zeros_like(noisyspec)
    posteri = np.zeros_like(noisyspec)
    for ii, xspec in enumerate(noisyspec):
        xpsd = np.abs(xspec)**2
        posteri[ii, :] = xpsd / npsd
        posteri_prime = np.maximum(posteri[ii, :] - 1, 0)  # half-wave rectify
        if ii == 0:  # initialize priori SNR
            priori[ii, :] = snrsmooth + (1-snrsmooth)*posteri_prime
        else:
            priori[ii, :] = snrsmooth*(
                priori2filt(priori[ii-1, :], rule)**2)\
                * posteri[ii-1, :] + (1-snrsmooth)*posteri_prime
        # compute speech presence log likelihood
        llk = posteri[ii, :]*priori[ii, :] / \
            (1+priori[ii, :]) - np.log(1+priori[ii, :])
        vad.append(np.mean(llk))
        if vad[ii] < llkthres:
            # noise only frame found, update Pn
            npsd = noisesmooth*npsd + (1-noisesmooth)*xpsd

        filt = priori2filt(priori[ii, :], rule=rule)  # compute gain
        # filter and store cleaned frame
        xfilt[ii, :] = filt*xspec

    return xfilt, priori, posteri, vad


def asnr_activate(x, sr, wind, hop, nfft, noise=None, zphase=True,
                  noisesmooth=.98, llkthres=.15, rule='wiener', fn='classic'):
    """Implement a priori SNR Estimation in the view of nonlinear activation.

    This implementation should be identital to asnr, except the estimation of
    mu is interpreted as an activation function. The activation function could
    be one of the following:
        * classic  - hard thresholding
        * step     - frequency-dependent hard thresholding
        * linear   - frequency-dependent piecewise linear
        * logistic - frequency-dependent logistic function
    """
    eps = 1e-16

    def activate(log_sigma_k, option):
        """VAD thresholding interpreted as an activation function."""
        mu = np.zeros_like(log_sigma_k)
        vad = np.mean(log_sigma_k) > llkthres
        if option == 'classic':
            # classic hard thresholding in Loizou
            if vad:
                mu[:] = 1
            else:
                mu[:] = noisesmooth
        elif option == 'step':
            # create frequency-dependent mu but each cell is the old
            # hard thresholding function
            mu[log_sigma_k >= llkthres] = 1
            mu[log_sigma_k < llkthres] = noisesmooth
        elif option == 'linear':
            # frequency-dependent mu again, but instead of a hard thresholding
            # function, each one is a piecewise linear function with sharp
            # gradient
            delta = .01  # delta in piecewise linear function activation
            offset = llkthres - delta
            k = (1-noisesmooth) / (2*delta)  # gradient of linear function
            mu = np.minimum(1, np.maximum(0, k*(
                log_sigma_k-offset)+noisesmooth))
        elif option == 'logistic':
            # frequency-dependent mu again, but use a logistic (sigmoidal)
            # function rather than linear function this time.
            mu = noisesmooth + (1-noisesmooth) / (
                1+np.exp(-(log_sigma_k-llkthres)))
        else:
            raise ValueError("option not supported!")

        return mu, vad

    # Estimate noise PSD first
    if noise is None:
        npsd_init = stpsd(x, wind, hop, nfft, nframes=6)
    else:
        npsd_init = stpsd(noise, wind, hop, nfft, nframes=-1)

    xfilt = []  # holds Wiener-filtered output
    vad = []  # holds voice activity decision

    # nonlinear activation for alpha
    E = np.eye(len(npsd_init))
    V = np.eye(len(npsd_init))
    bE = np.zeros_like(npsd_init)
    bV = np.zeros_like(npsd_init)
    # recurrent block
    xpsd_m1 = np.zeros_like(npsd_init)
    posteri_m1 = np.zeros_like(npsd_init)
    priori_m1 = np.zeros_like(npsd_init)
    llkr_m1 = np.zeros_like(npsd_init)
    for nn, xspec in enumerate(stft(x, wind, hop, nfft, synth=True,
                                    zphase=zphase)):
        if nn == 0:  # initial state
            xmag_m1 = np.abs(xspec)
            xpsd_m1 = xmag_m1**2
            npsd_m1 = npsd_init
            posteri_m1 = xpsd_m1/(npsd_m1+eps)
            priori_m1 = .98+(1-.98)*np.maximum(posteri_m1-1, 0)
            llkr_m1 = 1 - np.log(1+priori_m1)
        xmag = np.abs(xspec)
        xpsd = xmag ** 2
        mu, vad = activate(llkr_m1, fn)
        posteri = (xpsd / xpsd_m1) * posteri_m1 * (1/(mu+(1-mu)*posteri_m1))
        posteri_prime = np.maximum(posteri-1, 0)
        alpha_hat = 1/(1+((posteri_prime-priori_m1)/(posteri_prime+1))**2)

        z = np.maximum(0, E.dot(alpha_hat)+bE)
        alpha = np.minimum(1, np.maximum(0, V.dot(z)+bV))
        priori = alpha * (priori2filt(priori_m1, rule)**2)*posteri_m1 +\
            (1-alpha)*posteri_prime
        llkr = posteri * priori / (1+priori) - np.log(1+priori)
        filt = priori2filt(priori, rule)

        # update old values
        xpsd_m1[:] = xpsd
        posteri_m1[:] = posteri
        priori_m1[:] = priori
        llkr_m1[:] = llkr

        xfilt.append(filt*xspec)

    # Synthesis by overlap-add (OLA)
    xout = istft(xfilt, wind, hop, nfft, zphase=zphase)

    return xout


def asnr_recurrent(x, sr, wind, hop, nfft, noise=None, zphase=True,
                   snrsmooth=.98, noisesmooth=.98, llkthres=.15,
                   rule='wiener'):
    """Implement a priori SNR Estimation in a recurrent view.

    This implementation should be identital to `asnr`, except that the
    mask is estimated in a recurrent way. This implementation is close to the
    SNRNN.
    """
    # Estimate noise PSD first
    if noise is None:
        npsd_init = stpsd(x, wind, hop, nfft, nframes=6)
    else:
        npsd_init = stpsd(noise, wind, hop, nfft, nframes=-1)

    xfilt = []  # holds Wiener-filtered output

    freq_dim = nfft//2+1
    M = np.ones(freq_dim) * noisesmooth  # default mu value in Loizou
    K = np.eye(freq_dim)
    Eta = np.ones(freq_dim) * llkthres  # default vad threshold
    E = np.eye(freq_dim)
    V = np.eye(freq_dim)
    bE = np.zeros(freq_dim)
    bV = np.zeros(freq_dim)

    def forward(xmag_m1, xmag, posteri_m1, priori_m1, llkr_m1):

        # G1: Estimate mu(t) from lk_ratio(t-1)
        mu_t = M + (1-M) / (1+np.exp(-K.dot(llkr_m1-Eta)))

        # a posteriori SNR Estimation
        npsd_m1_over_npsd = 1/(mu_t + (1-mu_t)*posteri_m1)
        xpsd_m1 = xmag_m1 ** 2  # previous noisy PSD
        xpsd = xmag ** 2  # current noisy PSD
        posteri = npsd_m1_over_npsd * posteri_m1 * (xpsd/(xpsd_m1))

        posteri_prime = np.maximum(posteri - 1, 0)  # prevent negative
        alpha_hat = 1 / \
            (1+((posteri_prime-priori_m1)/(posteri_prime+1))**2)

        # G2: Estimate alpha(t) from alpha_hat(t)
        z = np.maximum(E.dot(alpha_hat) + bE, 0)
        # Clipping
        alpha = np.minimum(np.maximum(0, V.dot(z) + bV), 1)

        # A Priori SNR Estimation
        priori = alpha*(priori2filt(priori_m1, rule)**2)*posteri_m1 + \
            (1-alpha)*posteri_prime

        # Finally, calculate likelihood ratio of the current frame and
        # the gain function of the current frame, and the output frame
        filt = priori2filt(priori, rule=rule)  # Wiener optimal solution
        llkr = priori*posteri/(1+priori) - np.log(1+priori)
        print(np.mean(llkr) >= llkthres)

        return filt, posteri, priori, llkr

    # recurrent block
    xmag_m1 = np.zeros_like(npsd_init)
    posteri_m1 = np.zeros_like(npsd_init)
    priori_m1 = np.zeros_like(npsd_init)
    llkr_m1 = np.zeros_like(npsd_init)
    for nn, xspec in enumerate(stft(x, wind, hop, nfft, synth=True,
                                    zphase=zphase)):
        if nn == 0:
            # inital state
            """
            xmag_m1[:] = np.abs(xspec)
            xpsd_m1[:] = xmag_m1 ** 2
            npsd_m1[:] = npsd_init
            posteri_m1[:] = xpsd_m1/(npsd_m1+eps)
            priori_m1[:] = snrsmooth + \
                (1-snrsmooth)*np.maximum(posteri_m1-1, 0)
            llkr_m1[:] = 1 - np.log(1+priori_m1)
            """
            xmag = np.abs(xspec)
            xpsd = xmag**2
            posteri = xpsd / (npsd_init)
            priori = snrsmooth + (1-snrsmooth)*np.maximum(posteri - 1, 0)
            llkr = 1 - np.log(1+priori)
            filt = priori2filt(priori, rule)
        else:
            xmag = np.abs(xspec)
            filt, posteri, priori, llkr =\
                forward(xmag_m1, xmag, posteri_m1, priori_m1, llkr_m1)

        xfilt.append(filt*xspec)

        # update old values
        xmag_m1[:] = xmag
        posteri_m1[:] = posteri
        priori_m1[:] = priori
        llkr_m1[:] = llkr

    # Synthesis by overlap-add (OLA)
    xout = istft(xfilt, wind, hop, nfft, zphase=zphase)

    return xout


def mmse_henriks(sig, sr, wind, hop, nfft, noise=None, alpha=.98, beta=.8,
                 rule='wiener'):
    """Implement the MMSE method by Henriks et al for noise PSD estimation.

    Hendriks, Richard C., Richard Heusdens, and Jesper Jensen.
    "MMSE based noise PSD tracking with low complexity."
    2010 IEEE International Conference on Acoustics, Speech and Signal
    Processing. IEEE, 2010.
    Retrieved from http://cas.et.tudelft.nl/pubs/0004266.pdf

    Parameters
    ----------
    sig: array_like
        Noisy speech signal to be processed.
    sr: int
        Sampling rate.
    wind: array_like
        Window function for analysis.
    nfft: int
        Number of DFT points.
    alpha: float, optional
        Recursive averaging coefficient for decision-directed a priori SNR.
        Default to .98.
    beta: float, optional
        Recursive averaging coefficient for noise PSD.
        Default to .8, as suggested by Henriks et al.
    noise: array_like, optional
        Optional examplary noise signal from which noise PSD is extracted.
    rule: str, optional
        A priori SNR-to-spectral gain rule. Default to 'wiener'.
        Options are: wiener, specsub, ml

    See Also
    --------
    noise.mmse_henriks

    """
    sigspec = stft(sig, wind, hop, nfft, synth=True, zphase=True)
    _, priori, _ = npsd_henriks(sigspec.real**2+sigspec.imag**2, noise,
                                alpha, beta, rule)

    irm = priori2filt(priori, rule)
    xout = istft(sigspec*irm, wind, hop, nfft, zphase=True)

    return xout


def asnr_optim(x, t, sr, wind, hop, nfft, zphase=True, rule='wiener'):
    """Compute oracle mask for magnitude spectrogram."""
    eps = 1e-16
    siglen = min(len(x), len(t))
    if len(x) > siglen:
        x = x[:siglen]
    if len(t) > siglen:
        t = t[:siglen]

    xfilt = []
    for xspec, tspec in zip(
            stft(x, wind, hop, nfft, synth=True, zphase=zphase),
            stft(t, wind, hop, nfft, synth=True, zphase=zphase)):
        nspec = xspec - tspec
        tpsd = np.abs(tspec) ** 2
        npsd = np.abs(nspec) ** 2

        priori = tpsd / (npsd+eps)

        filt = priori2filt(priori, rule)
        xfilt.append(filt * xspec)

    # Synthesis by overlap-add (OLA)
    xout = istft(xfilt, wind, hop, nfft, zphase=zphase)

    return xout


def priori2filt(priori, rule='wiener'):
    """Compute filter gains given a priori SNR."""
    if rule == 'wiener':
        return priori / (1+priori)
    elif rule == 'specsub':
        return np.sqrt(priori / (1+priori))
    elif rule == 'ml':
        return 0.5*(1 + np.sqrt(priori / (1+priori)))
    else:
        raise NotImplementedError(
            "Suppression rules must be [wiener/specsub/ml]")
