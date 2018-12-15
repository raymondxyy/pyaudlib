"""[Plot]ting functions for feature visualization."""
import matplotlib.pyplot as plt
import numpy as np


def magresp(freq, resp, ax, units=('rad', 'db')):
    """Plot the magnitude response from complex frequency response.

    Parameters
    ----------
    freq : array_like
        a vector representing the x-axis
    resp : array_like
        complex frequency response
    ax : axis
        a matplotlib axis handle
    units : tuple of str, optional
        specify plotting units for x-axis and y-axis, respectively.
        Avalable options for x-axis:
            * 'rad' (discrete frequency in radians/sample, default)
            * 'hz' (physical frequency in Hz)
        Available options for y-axis:
            * 'db' (20*log10(|mag|), default)
            * 'mag' (absolute magnitude)

    """
    fu, ru = units
    mag = np.abs(resp)
    if ru == 'db':
        ax.plot(freq, 20*np.log10(mag+1e-16), 'b')
        ax.set_ylabel('Amplitude [dB]', color='b')
        ax.set_ylim(-40, max(10, np.max(mag)))
    else:
        ax.plot(freq, mag, 'b')
        ax.set_ylabel('Amplitude', color='b')
        ax.set_ylim(0, 1.25*np.max(mag))
    if fu == 'rad':
        ax.set_xlabel(r'Normalized Frequency [$\times \pi$ rad/sample]')
    else:
        ax.set_xlabel(r'Frequency [Hz]')
    ax.set_xlim(freq[0], freq[-1])
    ax.grid()


def phsresp(freq, resp, ax):
    ax.plot(freq, np.angle(resp)/np.pi, 'g')
    ax.set_ylabel(r'Angle ($\times \pi$ radians)', color='g')
    ax.set_xlabel(r'Normalized Frequency [$\times \pi$ rad/sample]')
    ax.set_ylim(-1, 1)
    ax.set_xlim(0, 1)
    ax.grid()


def cepline(qindex, cepstrum, ax, sr=None):
    """Draw a 1-D cepstrum line on an axis."""
    if sr is not None:
        qaxis = qindex/sr
        xlabel = 'Quefrency (s)'
    else:
        qaxis = qindex
        xlabel = 'Quefrency (sample)'
    line, = ax.plot(qaxis, cepstrum)
    ax.axis('tight')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Amplitude')
    return line


def plot_cep(qindex, cepstrum, sr=None, show=True, save=None):
    """Plot a cepstrum function in a figure.

    Arguments
    ---------
    qindex: int array
        Quefrency integer index in an array.
    cepstrum: float array
        Cepstrum in an array.
    sr: int/float [None]
        Sampling rate. If None, plot quefrency in integer index, otherwise plot
        in number of seconds.
    show: boolean [True]
        Show the figure?
    save: string [None]
        Save figure to file if exists.
    """
    fig, ax = plt.subplots()
    cepline(qindex, cepstrum, ax, sr=sr)
    if show:
        fig.show()
    if save is not None:
        fig.savefig(save)


def specgram(s, spec_xscale='linear', spec_yscale='linear'):
    """
    Plot specrogram given the spectrogram object.
    Args: s  - spectrogram object (t,f,m)
    """
    t, f, m = s
    fig = plt.figure()
    plt.pcolormesh(t, f, m.T, cmap='jet')
    #plt.pcolormesh(20*np.log10(m+eps),cmap='jet')
    plt.xscale(spec_xscale)
    plt.yscale(spec_yscale)
    plt.colorbar()
    plt.axis('tight')
    plt.xlabel("time (s)")
    plt.ylabel("frequency (hz)")
    plt.show()
