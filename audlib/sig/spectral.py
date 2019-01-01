"""Frame-level frequency-domain processing."""

import numpy as np
from numpy.fft import rfft, irfft


def magphasor(complexspec):
    """Decompose a complex spectrogram into magnitude and unit phasor.

    m, p = magphasor(c) such that c == m * p.
    """
    mspec = np.abs(complexspec)
    pspec = np.empty_like(complexspec)
    zero_mag = (mspec == 0.)  # fix zero-magnitude
    pspec[zero_mag] = 1.
    pspec[~zero_mag] = complexspec[~zero_mag]/mspec[~zero_mag]
    return mspec, pspec


def magphase(cspectrum, unwrap=False):
    """Decompose complex spectrum into magnitude and phase."""
    mag = np.abs(cspectrum)
    phs = np.angle(cspectrum)
    if unwrap:
        phs = np.unwrap(phs)
    return mag, phs


def logmagphase(cspectrum, unwrap=False, floor=-10.):
    """Compute (log-magnitude, phase) of complex spectrum."""
    mag, phs = magphase(cspectrum, unwrap=unwrap)
    return logmag(mag, floor=floor), phs


def logmag(sig, floor=-10.):
    """Compute log magnitude of complex spectrum.

    Floor any -`np.inf` value to `floor` plus log minimum. If all values are
    0s, floor all values to floor*5.
    """
    mag = np.abs(sig)
    zeros = mag == 0
    logm = np.empty_like(mag)
    logm[~zeros] = np.log(mag[~zeros])
    logmin = np.log(mag[~zeros].min()) if np.any(~zeros) else floor*5
    logm[zeros] = floor + logmin

    return logm


def phasor(mag, phase):
    """Compute complex spectrum given magnitude and phase."""
    return mag * np.exp(1j*phase)


def dct1(x, dft=False):
    """Perform Type-1 Discrete Cosine Transform (DCT-1) on input signal.

    Parameters
    ----------
    x: array_like
        Signal to be processed.
    dft: boolean
        Implement using dft?

    Returns
    -------
    X: array_like
        Type-1 DCT of x.

    """
    if len(x) == 1:
        return x.copy()
    ndct = len(x)

    if dft:  # implement using dft
        x_ext = np.concatenate((x, x[-2:0:-1]))  # create extended sequence
        X = np.real(rfft(x_ext)[:ndct])
    else:  # implement using definition
        xa = x * 1.
        xa[1:-1] *= 2.  # form x_a sequence
        X = np.zeros_like(xa)
        ns = np.arange(ndct)
        for k in range(ndct):
            cos = np.cos(np.pi*k*ns/(ndct-1))
            X[k] = cos.dot(xa)
    return X


def idct1(x_dct1, dft=False):
    """Perform inverse Type-1 Discrete Cosine Transform (iDCT-1) on spectrum.

    Parameters
    ----------
    x_dct1: array_like
        Input DCT spectrum.
    dft: boolean
        Implement using dft?

    Returns
    -------
    x: array_like
        Inverse Type-1 DCT of x_dct1.

    """
    if len(x_dct1) == 1:
        return x_dct1.copy()
    ndct = len(x_dct1)

    if dft:  # implement using dft
        x = irfft(x_dct1, n=2*(ndct-1))[:ndct]
    else:  # implement using definition
        Xb = x_dct1 / (ndct-1.)
        Xb[0] /= 2.
        Xb[-1] /= 2.
        x = np.zeros_like(Xb)
        ks = np.arange(ndct)
        for n in range(ndct):
            cos = np.cos(np.pi*n*ks/(ndct-1))
            x[n] = cos.dot(Xb)
    return x


def dct2(x, norm=True, dft=False):
    """Perform Type-2 Discrete Cosine Transform (DCT-2) on input signal.

    Parameters
    ----------
    x: array_like
        Input signal.
    norm: boolean, optional
        Normalize so that energy is preserved. Default to True.
    dft: boolean, optional
        Implement using dft? Default to False.

    Returns
    -------
    X: numpy array
        Type-2 DCT of x.

    """
    if len(x) == 1:
        return x.copy()
    ndct = len(x)

    if dft:  # implement using dft
        if norm:
            raise ValueError("DFT method does not support normalization!")
        Xk = rfft(x, 2*ndct)[:ndct]
        X = 2*np.real(Xk*np.exp(-1j*(np.pi*np.arange(ndct)/(2*ndct))))
    else:  # implement using definition
        if norm:
            xa = 1.*x
        else:
            xa = 2.*x
        X = np.zeros_like(xa)
        ns = np.arange(ndct)
        for k in range(ndct):
            cos = np.cos(np.pi*k*(2*ns+1)/(2*ndct))
            X[k] = cos.dot(xa)
            if norm:
                X[k] *= np.sqrt(2./ndct)
        if norm:
            X[0] /= np.sqrt(2)
    return X


def idct2(x_dct2, norm=True, dft=False):
    """Perform inverse Type-2 Discrete Cosine Transform (DCT-2) on spectrum.

    Parameters
    ----------
    x_dct2: array_like
        Input signal.
    norm: boolean, optional
        Normalize so that energy is preserved. Default to True.
    dft: boolean, optional
        Implement using dft? Default to False.

    Returns
    -------
    x: array_like
        Inverse Type-2 DCT of x_dct2.

    """
    if len(x_dct2) == 1:
        return x_dct2.copy()
    ndct = len(x_dct2)

    if dft:  # implement using dft
        if norm:
            raise ValueError("DFT method does not support normalization!")
        ks = np.arange(ndct)
        Xseg1 = x_dct2*np.exp(1j*np.pi*ks/(2*ndct))
        Xseg2 = -x_dct2[-1:0:-1]*np.exp(1j*np.pi*(ks[1:]+ndct)/(2*ndct))
        X_ext = np.concatenate((Xseg1, [0.], Xseg2))
        x = irfft(X_ext[:ndct+1])[:ndct]
    else:  # implement using definition
        if norm:
            Xb = x_dct2 * np.sqrt(2./ndct)
            Xb[0] /= np.sqrt(2.)
        else:
            Xb = x_dct2 / (ndct+0.0)
            Xb[0] /= 2.
        x = np.zeros_like(Xb)
        ks = np.arange(ndct)
        for n in range(ndct):
            cos = np.cos(np.pi*ks*(2*n+1)/(2*ndct))
            x[n] = cos.dot(Xb)
    return x


def realcep(frame, n, nfft=4096, floor=-10., comp=False, ztrans=False):
    """Compute real cepstrum of short-time signal `frame`.

    There are two modes for calculation:
        1. complex = False (default). This calculates c[n] using inverse DFT
        of the log magnitude spectrum.
        2. complex = True. This first calculates the complex cepstrum through
        the z-transform method (see `compcep`), and takes the even function to
        obtain c[n].
    In both cases, `len(cep) = nfft//2+1`.

    Parameters
    ----------
    frame: 1-D ndarray
        signal to be processed.
    nfft: non-negative int
        nfft//2+1 cepstrum in range [0, nfft//2] will be evaluated.
    floor: float [-10.]
        flooring for log(0). Ignored if complex=True.
    complex: boolean [False]
        Use mode 2 for calculation.

    Returns
    -------
    cep: 1-D ndarray
        Real ceptra of signal `frame` of:
        1. length `len(cep) = nfft//2+1`.
        2. quefrency index [0, nfft//2].

    """
    if comp:  # do complex method
        ccep = compcep(frame, n-1, ztrans=ztrans)
        rcep = .5*(ccep+ccep[::-1])
        return rcep[n-1:]  # only keep non-negative quefrency
    else:  # DFT method
        rcep = irfft(logmag(rfft(frame, nfft), floor=floor))
        return rcep[:n]


def compcep(frame, n, nfft=4096, floor=-10., ztrans=False):
    """Compute complex cepstrum of short-time signal using Z-transform.

    Compute the aliasing-free complex cepstrum using Z-transform and polynomial
    root finder. Implementation is based on RS eq 8.68 on page 436.

    Parameters
    ----------
    frame: 1-D ndarray
        signal to be processed.
    n: non-negative int
        index range [-n, n] in which complex cepstrum will be evaluated.

    Returns
    -------
    cep: 1-D ndarray
        complex ceptrum of length `2n+1`; quefrency index [-n, n].

    """
    if ztrans:
        frame = np.trim_zeros(frame)
        f0 = frame[0]
        roots = np.roots(frame/f0)
        rmag = np.abs(roots)
        assert 1 not in rmag
        ra, rb = roots[rmag < 1], roots[rmag > 1]
        amp = f0 * np.prod(rb)
        if len(rb) % 2:  # odd number of zeros outside UC
            amp = -amp
        # obtain complex cepstrum through eq (8.68) in RS, pp. 436
        cep = np.zeros(2*n+1)
        if rb.size > 0:
            for ii in range(-n, 0):
                cep[-n+ii] = np.real(np.sum(rb**ii))/ii
        cep[n] = np.log(np.abs(amp))
        if ra.size > 0:
            for ii in range(1, n+1):
                cep[n+ii] = -np.real(np.sum(ra**ii))/ii
    else:
        assert n <= nfft//2
        spec = rfft(frame, n=nfft)
        lmag, phase = logmagphase(spec, unwrap=True, floor=floor)
        cep = irfft(lmag+1j*phase, n=nfft)[:2*n+1]
        cep = np.roll(cep, n)
    return cep
