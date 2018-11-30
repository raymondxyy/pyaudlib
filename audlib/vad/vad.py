# A Voice activity detector based on a-priori SNR estimation
# Author: Raymond Xia (yangyanx@andrew.cmu.edu)

# Change log:
#   12/13/17:
#       * Create this file
#       * Added voice activity detection based on a-priori SNR estimation
#
from audlib.sig.stproc import stana
import numpy as np
from audlib.sig.time_analysis import *
from pprint import pprint as pp

CONFIG_VAD = {
    # STFT Parameters
    'SAMPLE_RATE': 16000,  # in Hz
    'WINDOW_LENGTH': 0.032,  # in seconds
    'FFT_SIZE': 512,  # in number of samples
    'HOP_FRACTION': 0.25,
    'ZERO_PHASE_STFT': True,  # use new zero-phase STFT
    # Parameters for A-priori SNR estimation method
    'ASNR_ALPHA': .98,  # smoothing factor for a-priori SNR
    'ASNR_MU': .98,  # smoothing factor for noise estimate
    'ASNR_VAD': .15,  # VAD threshold
    'ASNR_GLOBAL_VAD': .25  # global vad threshold
}


def asnr_vad(x, noise=None, vad_threshold=.15, verbose=False):
    """
    Output VAD decision instead of Wiener filtered speech.
    """
    # Estimate noise PSD first
    Pn = np.zeros(CONFIG_VAD['FFT_SIZE']/2+1)
    if noise is not None:  # User provide noise data
        for i, nframe in enumerate(stft_gen(noise, CONFIG_VAD['SAMPLE_RATE'],
                                            CONFIG_VAD['WINDOW_LENGTH'],
                                            CONFIG_VAD['HOP_FRACTION'],
                                            nfft=CONFIG_VAD['FFT_SIZE'],
                                            zero_phase=CONFIG_VAD['ZERO_PHASE_STFT'])):
            Pn += np.abs(nframe)**2  # collect PSD of all frames
        Pn /= (i+1)  # average over all frames
    else:  # default to first 6 frames of x
        for i, nframe in enumerate(stft_gen(x, CONFIG_VAD['SAMPLE_RATE'],
                                            CONFIG_VAD['WINDOW_LENGTH'],
                                            CONFIG_VAD['HOP_FRACTION'],
                                            nfft=CONFIG_VAD['FFT_SIZE'],
                                            zero_phase=CONFIG_VAD['ZERO_PHASE_STFT'])):
            Pn += np.abs(nframe)**2  # collect PSD of all frames
            if i == 5:
                break  # only average 6 frames
        Pn /= (i+1)  # average over all frames

    # Now perform frame-level Wiener filtering on x
    x_stft = ShortTimeAnalysis(x, CONFIG_VAD['SAMPLE_RATE'],
                               CONFIG_VAD['WINDOW_LENGTH'],
                               CONFIG_VAD['HOP_FRACTION'])
    vad = []  # holds voice activity decision
    for i, xframe in enumerate(x_stft):
        X = stft_frame(xframe, nfft=CONFIG_VAD['FFT_SIZE'],
                       zero_phase=CONFIG_VAD['ZERO_PHASE_STFT'])
        Px = np.abs(X)**2
        posteri = Px / Pn
        posteri_prime = np.maximum(posteri - 1, 0)  # half-wave rectify
        if i == 0:  # initialize priori SNR
            priori = CONFIG_VAD['ASNR_ALPHA'] +\
                    (1-CONFIG_VAD['ASNR_ALPHA'])*posteri_prime
        else:
            priori = CONFIG_VAD['ASNR_ALPHA']*(H_prev**2)*posteri_prev +\
                    (1-CONFIG_VAD['ASNR_ALPHA'])*posteri_prime
        # compute speech presence log likelihood
        llk = posteri*priori/(1+priori) - np.log(1+priori)
        vad.append((np.sum(llk)/xframe.size >= CONFIG_VAD['ASNR_VAD']))
        if not vad[i]:
            # noise only frame found, update Pn
            Pn = CONFIG_VAD['ASNR_MU']*Pn + (1-CONFIG_VAD['ASNR_MU'])*Px

        H = np.sqrt(priori / (1+priori))  # compute gain
        # update old values
        H_prev = H.copy()
        posteri_prev = posteri.copy()
    if verbose:
        print('+'*85)
        print('Frame-level decisions (20 frames per line):')
        for i in xrange(0, len(vad), 20):
            print(vad[i:min(i+20, len(vad))])
        print('+'*85)
    decision = (np.sum(vad)*1.0/len(vad) >= vad_threshold)
    if verbose:
        print('FINAL DECISION: [{}]'.format(decision))
    return decision


######### Use as standalone application ############
if __name__ == '__main__':
    import argparse
    from audio_io import audioread, audiowrite

    parser = argparse.ArgumentParser()
    parser.add_argument('-thres', help='in range [0,1]',
                        required=False, type=float, default=.15)
    parser.add_argument('-i', help='Input file', required=True)
    parser.add_argument('-n', help='Noise file', required=False)
    parser.add_argument('-v', help='Verbose', required=False,
                        action='store_true', default=False)
    args = parser.parse_args()

    # Processing block
    x, sr = audioread(args.i, sr=CONFIG_VAD['SAMPLE_RATE'],
                      force_mono=True, verbose=args.v)
    # Get rid of leading zeros
    x = np.trim_zeros(x, trim='f')
    if args.n:
        noise, sr = audioread(args.n, sr=CONFIG_VAD['SAMPLE_RATE'],
                              force_mono=True, verbose=args.v)
    else:
        noise = None

    asnr_vad(x, noise=noise, vad_threshold=args.thres, verbose=args.v)
