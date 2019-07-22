import numpy as np

from audlib.sig.transform import stft, istft, realcep, compcep
from audlib.io.audio import audioread
from audlib.sig.window import hamming

from audlib.sig.stproc import stana

sig, sr = audioread('samples/welcome16k.wav')
window_length = 0.05
hopfrac = 0.2
wind = hamming(int(window_length*sr), hop=hopfrac, synth=True)
nfft = 1024


def test_stft():
    """Test STFT and iSTFT."""
    sig_stft = stft(sig, sr, wind, hopfrac, nfft, synth=True, zphase=True)
    sigsynth = istft(sig_stft, sr, wind, hopfrac, nfft, zphase=True)

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
    ratcep = compcep(x, cepsize-1, ztrans=True)
    dftcep = compcep(x, cepsize-1, ztrans=False)
    assert np.allclose(cepref, ratcep[cepsize-1:])
    assert np.allclose(cepref, dftcep[cepsize-1:])
    # test real cepstrum
    cepref /= 2  # complex cepstrum is right-sided
    rcep1 = realcep(x, cepsize, ztrans=True)
    rcep2 = realcep(x, cepsize, ztrans=False)
    rcep3 = compcep(x, cepsize-1)
    rcep3 = .5*(rcep3[cepsize-1:]+rcep3[cepsize-1::-1])
    assert np.allclose(cepref, rcep1)
    assert np.allclose(cepref, rcep2)
    #assert np.allclose(cepref, rcep3)


def test_rcep():
    ncep = 500
    for frame in stana(sig[int(.652*sr):], sr, wind, hopfrac):
        cep1 = realcep(frame, ncep)  # log-magnitude method
        cep2 = realcep(frame, ncep, comp=True, ztrans=True)  # ZT method
        cep3 = realcep(frame, ncep, comp=True)  # complex log method
        #assert np.allclose(cep1, cep2)
        break


if __name__ == '__main__':
    test_stft()
