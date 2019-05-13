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
    if fu == 'rad':
        ax.set_xlabel(r'Normalized Frequency [$\times \pi$ rad/sample]')
    else:
        ax.set_xlabel(r'Frequency [Hz]')
    ax.set_xlim(freq[0], freq[-1])
    ax.grid()


def phsresp(freq, resp, ax):
    """Plot phase response from the complex frequency response.

    Parameters
    ----------
    freq: array_like
        Discrete frequency points
    resp: array_like
        Complex frequency response
    ax: matplotlib.Axes object
        axis to be plotted on

    """
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

    Parameters
    ----------
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


def specgram(sig, ax, time_axis=None, freq_axis=None, colorbar=True):
    """Plot a 2-D representation as a spectrogram.

    Warning: `sig` must be in the final numeric form (no complex numbers).

    Parameters
    ----------
    sig: array_like
        2-D signal to bee plotted as a spectrogram
    ax: matplotlib.Axes
        Axis on which the spectrogram is plotted
    time_axis: array_like, optional
        Must have same dimension as `stft.shape[0]`.
        Default will plot on sample scale.
    freq_axis: array_like, optional
        Must have same dimension as `stft.shape[1]`.
        Default will plot on number of channels.

    """
    if time_axis is None:
        time_axis = np.arange(sig.shape[0])
    if freq_axis is None:
        freq_axis = np.arange(sig.shape[1])

    spec = ax.pcolormesh(time_axis, freq_axis, sig.T, cmap='jet')
    if colorbar:
        plt.colorbar(spec, ax=ax)

    return
