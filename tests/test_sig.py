from audlib.sig.transform import stft
from audlib.sig.stproc import stft as stft_gen, istft as istft_gen
from audlib.io.audio import audioread
from audlib.sig.window import hamming, cola
import numpy as np
import audlib.sig.stproc as stproc
from audlib.plot import cepline
import matplotlib.pyplot as plt

sig, fs = audioread('samples/welcome16k.wav')
window_length = 0.032
hop_fraction = 0.25
wind = hamming(int(window_length*fs), hop_fraction)
nfft = 512


def test_stft():

    _, __, sig_stft = stft(sig, fs, window_length=window_length,
                           hop_fraction=hop_fraction)
    # STFT
    sig_stft_mag = np.abs(sig_stft)
    sig_stft_gen = np.array(list(stft_gen(sig, fs, wind, hop_fraction, nfft)))
    sig_stft_gen_mag = np.abs(sig_stft_gen)
    #assert np.allclose(sig_stft_mag, sig_stft_gen_mag)
    # iSTFT
    sigsynth = istft_gen(sig_stft_gen, fs, wind, hop_fraction, nfft)
    assert np.allclose(sig, sigsynth[:len(sig)])


def test_cep():
    """
    Test complex cepstrum function using the impulse response of an echo.
    Taken from example 8.3 of RS, page 414.
    """
    # the complex cepstrum of simple echo is an impulse train with decaying
    # magnitude.
    Np = 8
    alpha = .5  # echo impulse
    x = np.zeros(512)
    x[0], x[Np] = 1, alpha

    cepsize = 150
    # construct reference
    cepref = np.zeros(cepsize)
    for kk in range(1, cepsize // Np+1):
        cepref[kk*Np] = (-1)**(kk+1) * (alpha**kk)/kk

    # test complex cepstrum
    ratcep = stproc.compcep(x, cepsize-1, ztrans=True)
    dftcep = stproc.compcep(x, cepsize-1, ztrans=False)
    assert np.allclose(cepref, ratcep[cepsize-1:])
    assert np.allclose(cepref, dftcep[cepsize-1:])
    # test real cepstrum
    cepref /= 2  # complex cepstrum is right-sided
    rcep1 = stproc.realcep(x, cepsize, ztrans=True)
    rcep2 = stproc.realcep(x, cepsize, ztrans=False)
    rcep3 = stproc.compcep(x, cepsize-1)
    rcep3 = .5*(rcep3[cepsize-1:]+rcep3[cepsize-1::-1])
    assert np.allclose(cepref, rcep1)
    assert np.allclose(cepref, rcep2)
    #assert np.allclose(cepref, rcep3)


def test_rcep():
    ncep = 500
    for frame in stproc.stana(sig, fs, wind, hop_fraction, trange=(.652, None)):
        cep1 = stproc.realcep(frame, ncep)  # log-magnitude method
        cep2 = stproc.realcep(frame, ncep, comp=True, ztrans=True)  # ZT method
        cep3 = stproc.realcep(frame, ncep, comp=True)  # complex log method
        qindex = np.arange(ncep)[:]
        fig, ax = plt.subplots(3, 1)
        line1 = cepline(qindex, cep1[qindex], ax[0])
        line1.set_label('DFT Method: Real Logarithm')
        line2 = cepline(qindex, cep2[qindex], ax[1])
        line2.set_label('Z-Transform Method')
        line3 = cepline(qindex, cep3[qindex], ax[2])
        line3.set_label('DFT Method: Complex Logarithm')
        for axe in ax:
            axe.legend()
        fig.savefig('out.pdf')
        #assert np.allclose(cep1, cep2)
        break


if __name__ == '__main__':
    test_cep()
