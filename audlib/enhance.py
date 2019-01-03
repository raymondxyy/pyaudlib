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

from .sig.stproc import stana, ola
from .sig.temporal import lpc
from .sig.transform import stft, istft, stpsd
from .cfg import cfgload

# Load in configurations first
_cfg = cfgload()['enhance']
# LPC Parameters for iterative Wiener filtering method
_lpc_order = int(_cfg['lpc_order'])
_lpc_method = str(_cfg['lpc_method'])
# Parameters for A-priori SNR estimation method
_asnr_alpha = float(_cfg['asnr_alpha'])  # smoothing factor for a-priori SNR
_asnr_mu = float(_cfg['asnr_mu'])  # smoothing factor for noise estimate
_asnr_vad = float(_cfg['asnr_vad'])  # VAD threshold


def __cfgshow__():
    for field in _cfg:
        print("{:>15}: [{}]".format(field, _cfg[field]))


def wiener_iter(x, sr, wind, hop, nfft, noise=None, zphase=True, iters=3):
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
                        _lpc_order+1), np.arange(nfft//2+1)))

    # Basically copy from stft code to prevent redundant processing
    fsize = len(wind)
    woff = (fsize-(fsize % 2)) // 2
    zp = np.zeros(nfft-fsize)  # zero padding

    # Estimate noise PSD first
    if noise is None:
        npsd = stpsd(x, sr, wind, hop, nfft, nframes=6)
    else:
        npsd = stpsd(noise, sr, wind, hop, nfft, nframes=-1)

    def wiener_iter_frame(frame):
        """One frame of Wiener iterative filtering."""
        xlpc = lpc(frame, _lpc_order, _lpc_method)
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
            xlpc = lpc(buff, _lpc_order, _lpc_method)
        return buff, filt, spsd

    # Now perform frame-level Wiener filtering on x
    srec = []
    for xframe in stana(x, sr, wind, hop):
        xclean, filt, spsd = wiener_iter_frame(xframe)
        srec.append(xclean)

    # Synthesis by overlap-add (OLA)
    xsynth = ola(srec, sr, wind, hop)

    return xsynth


def asnr(x, sr, wind, hop, nfft, noise=None, zphase=True, rule='wiener'):
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
        npsd = stpsd(x, sr, wind, hop, nfft, nframes=6)
    else:
        npsd = stpsd(noise, sr, wind, hop, nfft, nframes=-1)

    xfilt = []  # holds Wiener-filtered output
    vad = []  # holds voice activity decision
    priori_m1 = np.zeros_like(npsd)
    posteri_m1 = np.zeros_like(npsd)
    for i, xspec in enumerate(stft(x, sr, wind, hop, nfft, zphase=zphase)):
        xpsd = np.abs(xspec)**2
        posteri = xpsd / npsd
        posteri_prime = np.maximum(posteri - 1, 0)  # half-wave rectify
        if i == 0:  # initialize priori SNR
            priori = _asnr_alpha + (1-_asnr_alpha)*posteri_prime
        else:
            priori = _asnr_alpha*(priori2filt(priori_m1, rule)**2) *\
                posteri_m1 + (1-_asnr_alpha)*posteri_prime
        # compute speech presence log likelihood
        llk = posteri*priori/(1+priori) - np.log(1+priori)
        vad.append(np.mean(llk))
        if vad[i] < _asnr_vad:
            # noise only frame found, update Pn
            npsd = _asnr_mu*npsd + (1-_asnr_mu)*xpsd

        filt = priori2filt(priori, rule=rule)  # compute gain
        # filter and store cleaned frame
        xfilt.append(filt*xspec)

        # update old values
        priori_m1[:] = priori
        posteri_m1[:] = posteri

    # Synthesis by overlap-add (OLA)
    xout = istft(xfilt, sr, wind, hop, nfft, zphase=zphase)

    return xout, vad


def asnr_spec(noisyspec, rule='wiener'):
    """Implement the a-priori SNR estimation described by Scalart and Filho.

    This is very similar t `asnr`, except it's computed directly on noisy
    spectra instead of time-domain signals.

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
            priori[ii, :] = _asnr_alpha + (1-_asnr_alpha)*posteri_prime
        else:
            priori[ii, :] = _asnr_alpha*(
                priori2filt(priori[ii-1, :], rule)**2)\
                * posteri[ii-1, :] + (1-_asnr_alpha)*posteri_prime
        # compute speech presence log likelihood
        llk = posteri[ii, :]*priori[ii, :] / \
            (1+priori[ii, :]) - np.log(1+priori[ii, :])
        vad.append(np.mean(llk))
        if vad[ii] < _asnr_vad:
            # noise only frame found, update Pn
            npsd = _asnr_mu*npsd + (1-_asnr_mu)*xpsd

        filt = priori2filt(priori[ii, :], rule=rule)  # compute gain
        # filter and store cleaned frame
        xfilt[ii, :] = filt*xspec

    return xfilt, priori, posteri, vad


def asnr_activate(x, sr, wind, hop, nfft, noise=None, zphase=True,
                  rule='wiener', fn='classic'):
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
        vad = np.mean(log_sigma_k) > _asnr_vad
        if option == 'classic':
            # classic hard thresholding in Loizou
            if vad:
                mu[:] = 1
            else:
                mu[:] = _asnr_mu
        elif option == 'step':
            # create frequency-dependent mu but each cell is the old
            # hard thresholding function
            mu[log_sigma_k >= _asnr_vad] = 1
            mu[log_sigma_k < _asnr_vad] = _asnr_mu
        elif option == 'linear':
            # frequency-dependent mu again, but instead of a hard thresholding
            # function, each one is a piecewise linear function with sharp
            # gradient
            delta = .01  # delta in piecewise linear function activation
            offset = _asnr_vad - delta
            k = (1-_asnr_mu) / (2*delta)  # gradient of linear function
            mu = np.minimum(1, np.maximum(0, k*(log_sigma_k-offset)+_asnr_mu))
        elif option == 'logistic':
            # frequency-dependent mu again, but use a logistic (sigmoidal)
            # function rather than linear function this time.
            mu = _asnr_mu + (1-_asnr_mu) / (1+np.exp(-(log_sigma_k-_asnr_vad)))
        else:
            raise ValueError("option not supported!")

        return mu, vad

    # Estimate noise PSD first
    if noise is None:
        npsd_init = stpsd(x, sr, wind, hop, nfft, nframes=6)
    else:
        npsd_init = stpsd(noise, sr, wind, hop, nfft, nframes=-1)

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
    for nn, xspec in enumerate(stft(x, sr, wind, hop, nfft, zphase=zphase)):
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
    xout = istft(xfilt, sr, wind, hop, nfft, zphase=zphase)

    return xout


def asnr_recurrent(x, sr, wind, hop, nfft, noise=None, zphase=True,
                   rule='wiener'):
    """Implement a priori SNR Estimation in a recurrent view.

    This implementation should be identital to `asnr`, except that the
    mask is estimated in a recurrent way. This implementation is close to the
    SNRNN.
    """
    # Estimate noise PSD first
    if noise is None:
        npsd_init = stpsd(x, sr, wind, hop, nfft, nframes=6)
    else:
        npsd_init = stpsd(noise, sr, wind, hop, nfft, nframes=-1)

    xfilt = []  # holds Wiener-filtered output

    freq_dim = nfft//2+1
    M = np.ones(freq_dim) * _asnr_mu  # default mu value in Loizou
    K = np.eye(freq_dim)
    Eta = np.ones(freq_dim) * _asnr_vad  # default vad threshold
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
        print(np.mean(llkr) >= _asnr_vad)

        return filt, posteri, priori, llkr

    # recurrent block
    xmag_m1 = np.zeros_like(npsd_init)
    posteri_m1 = np.zeros_like(npsd_init)
    priori_m1 = np.zeros_like(npsd_init)
    llkr_m1 = np.zeros_like(npsd_init)
    for nn, xspec in enumerate(stft(x, sr, wind, hop, nfft, zphase=zphase)):
        if nn == 0:
            # inital state
            """
            xmag_m1[:] = np.abs(xspec)
            xpsd_m1[:] = xmag_m1 ** 2
            npsd_m1[:] = npsd_init
            posteri_m1[:] = xpsd_m1/(npsd_m1+eps)
            priori_m1[:] = _asnr_alpha + \
                (1-_asnr_alpha)*np.maximum(posteri_m1-1, 0)
            llkr_m1[:] = 1 - np.log(1+priori_m1)
            """
            xmag = np.abs(xspec)
            xpsd = xmag**2
            posteri = xpsd / (npsd_init)
            priori = _asnr_alpha + (1-_asnr_alpha)*np.maximum(posteri - 1, 0)
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
    xout = istft(xfilt, sr, wind, hop, nfft, zphase=zphase)

    return xout


def asnr_optim(x, t, sr, wind, hop, nfft, noise=None, zphase=True,
               rule='wiener'):
    """Compute oracle mask for magnitude spectrogram."""
    eps = 1e-16
    siglen = min(len(x), len(t))
    if len(x) > siglen:
        x = x[:siglen]
    if len(t) > siglen:
        t = t[:siglen]

    xfilt = []
    for xspec, tspec in zip(stft(x, sr, wind, hop, nfft, zphase=zphase),
                            stft(t, sr, wind, hop, nfft, zphase=zphase)):
        nspec = xspec - tspec
        tpsd = np.abs(tspec) ** 2
        npsd = np.abs(nspec) ** 2

        priori = tpsd / (npsd+eps)

        filt = priori2filt(priori, rule)
        xfilt.append(filt * xspec)

    # Synthesis by overlap-add (OLA)
    xout = istft(xfilt, sr, wind, hop, nfft, zphase=zphase)

    return xout


def priori2filt(priori, rule):
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
